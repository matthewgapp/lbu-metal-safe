//! Narrow safe operations for a direct Metal presenter.
//!
//! The generated `objc2-metal` API is already safe for ordinary object operations. This crate owns
//! only the operations whose platform signatures necessarily contain raw window handles, raw byte
//! pointers, or Objective-C block pointers. It deliberately exposes no renderer, command graph,
//! scene, shader, pipeline, fallback, or API-selection abstraction.

use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::mpsc;

use block2::RcBlock;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLDrawable, MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLTexture,
};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle};
use raw_window_metal::Layer;

/// Failure to obtain a Metal layer from one live borrowed window.
#[derive(Debug)]
pub enum WindowLayerError {
    /// AppKit layer installation was attempted away from the main thread.
    NotMainThread,
    /// The window did not provide a live raw window handle.
    Handle(HandleError),
    /// The borrowed window is not an AppKit window.
    UnsupportedWindow,
}

/// Installs or obtains the `CAMetalLayer` owned by one borrowed AppKit window.
///
/// The returned layer is retained independently, while the window borrow proves the raw `NSView`
/// handle is live for the complete installation call. `raw-window-metal` installs a tracking
/// sublayer without replacing application window semantics.
#[cfg(target_os = "macos")]
pub fn layer_for_window(
    window: &impl HasWindowHandle,
) -> Result<Retained<CAMetalLayer>, WindowLayerError> {
    if MainThreadMarker::new().is_none() {
        return Err(WindowLayerError::NotMainThread);
    }
    let handle = window.window_handle().map_err(WindowLayerError::Handle)?;
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return Err(WindowLayerError::UnsupportedWindow);
    };
    // SAFETY: `WindowHandle<'_>` guarantees that `ns_view` is a valid live `NSView` for the
    // duration of this call. Main-thread affinity was checked above.
    let layer = unsafe { Layer::from_ns_view(handle.ns_view) };
    let pointer = layer.into_raw().cast::<CAMetalLayer>();
    // SAFETY: `Layer::into_raw` transfers one +1 retain count for an actual `CAMetalLayer`.
    Ok(unsafe { Retained::from_raw(pointer.as_ptr()) }
        .expect("raw-window-metal returned a non-null retained layer"))
}

/// Why a slice-qualified shared-buffer operation could not complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferTransferError {
    /// Metal refused to allocate the requested shared buffer.
    Allocation,
    /// The requested byte range does not fit the exact Metal buffer.
    Range,
}

/// Allocates one shared Metal buffer initialized from the exact supplied bytes.
pub fn shared_buffer_with_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, BufferTransferError> {
    if bytes.is_empty() {
        return device
            .newBufferWithLength_options(0, MTLResourceOptions::StorageModeShared)
            .ok_or(BufferTransferError::Allocation);
    }
    let pointer = NonNull::from(bytes).cast::<c_void>();
    // SAFETY: `pointer` addresses exactly `bytes.len()` readable bytes for the duration of this
    // synchronous copying constructor.
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(BufferTransferError::Allocation)
}

/// Copies exact bytes into an existing shared Metal buffer.
pub fn write_shared_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    bytes: &[u8],
) -> Result<(), BufferTransferError> {
    let end = offset
        .checked_add(bytes.len())
        .ok_or(BufferTransferError::Range)?;
    if end > buffer.length() {
        return Err(BufferTransferError::Range);
    }
    if bytes.is_empty() {
        return Ok(());
    }
    let destination = buffer.contents().cast::<u8>();
    // SAFETY: the range check above proves the exact destination interval lies within this shared
    // buffer; `bytes` supplies the exact readable source interval; the intervals cannot overlap
    // because Metal owns the destination allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            destination.as_ptr().add(offset),
            bytes.len(),
        )
    };
    Ok(())
}

/// Reads one exact byte range from an existing shared Metal buffer.
pub fn read_shared_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    range: Range<usize>,
) -> Result<Vec<u8>, BufferTransferError> {
    if range.start > range.end || range.end > buffer.length() {
        return Err(BufferTransferError::Range);
    }
    let mut bytes = vec![0; range.len()];
    if bytes.is_empty() {
        return Ok(bytes);
    }
    let source = buffer.contents().cast::<u8>();
    // SAFETY: the range check proves the source interval lies within this shared buffer, and the
    // newly allocated destination contains exactly `range.len()` writable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            source.as_ptr().add(range.start),
            bytes.as_mut_ptr(),
            bytes.len(),
        )
    };
    Ok(bytes)
}

/// Why an exact four-byte-per-pixel texture transfer could not complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureTransferError {
    /// The texture format is not one of the exact supported four-byte color encodings.
    UnsupportedFormat,
    /// The requested origin or extent lies outside the texture.
    Region,
    /// The byte slice length is not exactly four bytes per requested pixel.
    ByteCount,
}

fn supports_four_byte_color(format: MTLPixelFormat) -> bool {
    matches!(
        format,
        MTLPixelFormat::RGBA8Unorm
            | MTLPixelFormat::RGBA8Unorm_sRGB
            | MTLPixelFormat::BGRA8Unorm
            | MTLPixelFormat::BGRA8Unorm_sRGB
    )
}

fn checked_rgba8_region(
    texture: &ProtocolObject<dyn MTLTexture>,
    origin: [usize; 2],
    extent: [usize; 2],
    byte_len: usize,
) -> Result<(MTLRegion, usize), TextureTransferError> {
    if !supports_four_byte_color(texture.pixelFormat()) {
        return Err(TextureTransferError::UnsupportedFormat);
    }
    let end_x = origin[0]
        .checked_add(extent[0])
        .ok_or(TextureTransferError::Region)?;
    let end_y = origin[1]
        .checked_add(extent[1])
        .ok_or(TextureTransferError::Region)?;
    if extent.contains(&0) || end_x > texture.width() || end_y > texture.height() {
        return Err(TextureTransferError::Region);
    }
    let bytes_per_row = extent[0]
        .checked_mul(4)
        .ok_or(TextureTransferError::ByteCount)?;
    let required = bytes_per_row
        .checked_mul(extent[1])
        .ok_or(TextureTransferError::ByteCount)?;
    if byte_len != required {
        return Err(TextureTransferError::ByteCount);
    }
    Ok((
        MTLRegion {
            origin: objc2_metal::MTLOrigin {
                x: origin[0],
                y: origin[1],
                z: 0,
            },
            size: objc2_metal::MTLSize {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
        },
        bytes_per_row,
    ))
}

/// Replaces one exact two-dimensional four-byte-per-pixel texture region.
pub fn replace_texture_rgba8(
    texture: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    origin: [usize; 2],
    extent: [usize; 2],
    bytes: &[u8],
) -> Result<(), TextureTransferError> {
    let (region, bytes_per_row) = checked_rgba8_region(texture, origin, extent, bytes.len())?;
    let pointer = NonNull::from(bytes).cast::<c_void>();
    // SAFETY: validation above proves the pointer supplies the exact byte interval Metal reads for
    // this four-byte-per-pixel region and row stride during the synchronous copy.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            level,
            pointer,
            bytes_per_row,
        )
    };
    Ok(())
}

/// Reads one exact two-dimensional four-byte-per-pixel texture region.
pub fn read_texture_rgba8(
    texture: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    origin: [usize; 2],
    extent: [usize; 2],
) -> Result<Vec<u8>, TextureTransferError> {
    let byte_len = extent[0]
        .checked_mul(extent[1])
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextureTransferError::ByteCount)?;
    let (region, bytes_per_row) = checked_rgba8_region(texture, origin, extent, byte_len)?;
    let mut bytes = vec![0; byte_len];
    let pointer = NonNull::from(bytes.as_mut_slice()).cast::<c_void>();
    // SAFETY: validation above proves the destination supplies the exact writable byte interval
    // Metal fills for this four-byte-per-pixel region and row stride.
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(pointer, bytes_per_row, region, level)
    };
    Ok(bytes)
}

/// Terminal evidence reported for the exact drawable registered with Metal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresentedDrawable {
    drawable_id: usize,
    presented_time: f64,
}

impl PresentedDrawable {
    /// Exact monotonic drawable identity from its owning `CAMetalLayer`.
    pub const fn drawable_id(self) -> usize {
        self.drawable_id
    }

    /// Host time at which the exact drawable appeared on screen.
    pub const fn presented_time(self) -> f64 {
        self.presented_time
    }
}

/// Terminal failure reported for one registered drawable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentedDrawableError {
    /// The callback named a different object or drawable identity.
    Identity,
    /// Metal reported zero, negative, or non-finite presentation time.
    NotPresented,
    /// The completion channel ended without terminal drawable evidence.
    Cancelled,
    /// The caller polled a completion after already consuming its terminal result.
    AlreadyCompleted,
}

/// Progress of one exact registered Metal drawable.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "pending drawable presentation must continue to be polled"]
pub enum PresentedDrawableProgress {
    /// The post-presentation callback has not run yet.
    Pending,
    /// The exact registered drawable appeared on screen.
    Presented(PresentedDrawable),
}

type PresentedBlock = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLDrawable>>) + 'static>;

/// Owned one-shot completion for one exact registered Metal drawable.
pub struct PresentedDrawableCompletion {
    receiver: mpsc::Receiver<Result<PresentedDrawable, PresentedDrawableError>>,
    _block: PresentedBlock,
    completed: bool,
}

impl PresentedDrawableCompletion {
    /// Polls without blocking for the exact drawable's terminal presentation result.
    pub fn try_complete(&mut self) -> Result<PresentedDrawableProgress, PresentedDrawableError> {
        if self.completed {
            return Err(PresentedDrawableError::AlreadyCompleted);
        }
        match self.receiver.try_recv() {
            Ok(result) => {
                self.completed = true;
                result.map(PresentedDrawableProgress::Presented)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(PresentedDrawableProgress::Pending),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.completed = true;
                Err(PresentedDrawableError::Cancelled)
            }
        }
    }
}

/// Registers one owned post-presentation callback for the exact supplied drawable.
pub fn register_presented_drawable(
    drawable: &ProtocolObject<dyn MTLDrawable>,
) -> PresentedDrawableCompletion {
    let expected_address = NonNull::from(drawable).cast::<()>();
    let expected_id = drawable.drawableID();
    let (sender, receiver) = mpsc::sync_channel(1);
    let block = RcBlock::new(move |actual: NonNull<ProtocolObject<dyn MTLDrawable>>| {
        let actual_address = actual.cast::<()>();
        // SAFETY: Metal invokes `MTLDrawablePresentedHandler` with a valid drawable pointer
        // for the duration of this callback.
        let actual = unsafe { actual.as_ref() };
        let actual_id = actual.drawableID();
        let presented_time = actual.presentedTime();
        let result = if actual_address != expected_address || actual_id != expected_id {
            Err(PresentedDrawableError::Identity)
        } else if !presented_time.is_finite() || presented_time <= 0.0 {
            Err(PresentedDrawableError::NotPresented)
        } else {
            Ok(PresentedDrawable {
                drawable_id: actual_id,
                presented_time,
            })
        };
        let _ = sender.try_send(result);
    });
    // SAFETY: the heap-owned block remains alive in `PresentedDrawableCompletion`; its signature
    // exactly matches `MTLDrawablePresentedHandler`, and Metal copies/retains registered handlers.
    unsafe { drawable.addPresentedHandler(RcBlock::as_ptr(&block)) };
    PresentedDrawableCompletion {
        receiver,
        _block: block,
        completed: false,
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    use super::*;

    #[test]
    fn metal_shared_buffer_round_trip_is_exact_and_range_checked() {
        let device = MTLCreateSystemDefaultDevice().expect("test host has a Metal device");
        let initial = [1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = shared_buffer_with_bytes(&device, &initial).expect("shared allocation works");
        assert_eq!(
            read_shared_buffer(&buffer, 0..initial.len()).expect("whole buffer is readable"),
            initial
        );

        write_shared_buffer(&buffer, 2, &[9, 10, 11]).expect("bounded write works");
        assert_eq!(
            read_shared_buffer(&buffer, 0..initial.len()).expect("changed buffer is readable"),
            [1, 2, 9, 10, 11, 6, 7, 8]
        );
        assert_eq!(
            write_shared_buffer(&buffer, initial.len(), &[12]),
            Err(BufferTransferError::Range)
        );
        assert_eq!(
            read_shared_buffer(&buffer, 3..initial.len() + 1),
            Err(BufferTransferError::Range)
        );
    }
}
