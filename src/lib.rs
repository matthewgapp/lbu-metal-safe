//! Narrow safe operations for a direct Metal presenter.
//!
//! The generated `objc2-metal` API is already safe for ordinary object operations. This crate owns
//! only the operations whose platform signatures necessarily contain raw window handles, raw byte
//! pointers, Objective-C block pointers, or unchecked Metal argument indices. It deliberately
//! exposes no renderer, command graph, scene, shader, fallback, or API-selection abstraction.

mod render;

pub use render::{
    BoundVertexRecords, ColorLoad, ColorStore, PendingPresentedRender, PendingRender,
    PendingRenderProgress, RenderCommandBuffer, RenderCommandError, RenderExecutionStatus,
    RenderExecutionTiming, RenderPassDescriptor, RenderPipelineAttachmentError, Texture2DMips,
    Texture2DStorage, Texture2DUse, TextureAllocationError, new_texture_2d,
    render_pipeline_color_attachment,
};

use std::ffi::c_void;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock, mpsc};

use block2::RcBlock;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLDrawable, MTLPixelFormat, MTLRegion, MTLResource, MTLResourceOptions,
    MTLStorageMode, MTLTexture, MTLTextureType,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle};
use raw_window_metal::Layer;

/// Why one immutable drawable-generation identity could not be minted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawablePresentationGenerationError {
    /// The supplied extent was zero or differed from the exact layer drawable extent.
    Extent,
    /// The supplied external host generation was zero.
    Identity,
}

/// Opaque identity of one exact host-qualified Metal layer configuration.
///
/// Only the owning window/presenter observes every physical configuration mutation, so it supplies
/// the nonzero monotonic epoch. That epoch must advance for every mutation, including one that later
/// returns to an earlier extent. This boundary validates the retained layer, extent, drawable, and
/// exact equality of the host identity without pretending to own QuartzCore configuration.
#[derive(Clone)]
pub struct DrawablePresentationGeneration {
    layer: Retained<CAMetalLayer>,
    external_epoch: u64,
    width: u32,
    height: u32,
}

impl DrawablePresentationGeneration {
    /// Binds one retained layer and current extent to a nonzero host-owned physical epoch.
    pub fn try_new(
        layer: Retained<CAMetalLayer>,
        width: u32,
        height: u32,
        external_epoch: u64,
    ) -> Result<Self, DrawablePresentationGenerationError> {
        if external_epoch == 0 {
            return Err(DrawablePresentationGenerationError::Identity);
        }
        let size = layer.drawableSize();
        if width == 0
            || height == 0
            || size.width != f64::from(width)
            || size.height != f64::from(height)
        {
            return Err(DrawablePresentationGenerationError::Extent);
        }
        Ok(Self {
            layer,
            external_epoch,
            width,
            height,
        })
    }

    pub(crate) fn matches_drawable(&self, drawable: &ProtocolObject<dyn CAMetalDrawable>) -> bool {
        let layer = drawable.layer();
        let layer_ref: &CAMetalLayer = &layer;
        let retained_layer_ref: &CAMetalLayer = &self.layer;
        let size = layer.drawableSize();
        let texture = drawable.texture();
        std::ptr::eq(layer_ref, retained_layer_ref)
            && size.width == f64::from(self.width)
            && size.height == f64::from(self.height)
            && texture.width() == self.width as usize
            && texture.height() == self.height as usize
    }
}

impl std::fmt::Debug for DrawablePresentationGeneration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DrawablePresentationGeneration")
            .field("external_epoch", &self.external_epoch)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl PartialEq for DrawablePresentationGeneration {
    fn eq(&self, other: &Self) -> bool {
        let self_layer: &CAMetalLayer = &self.layer;
        let other_layer: &CAMetalLayer = &other.layer;
        std::ptr::eq(self_layer, other_layer)
            && self.external_epoch == other.external_epoch
            && self.width == other.width
            && self.height == other.height
    }
}

impl Eq for DrawablePresentationGeneration {}

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
    /// The supplied buffer is not CPU-accessible shared storage.
    UnsupportedStorage,
    /// The requested byte range does not fit the exact Metal buffer.
    Range,
}

fn require_shared_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), BufferTransferError> {
    if buffer.storageMode() != MTLStorageMode::Shared {
        return Err(BufferTransferError::UnsupportedStorage);
    }
    Ok(())
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
    require_shared_buffer(buffer)?;
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
    require_shared_buffer(buffer)?;
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

/// Why an exact supported texture transfer could not complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureTransferError {
    /// The supplied texture is not CPU-accessible shared storage.
    UnsupportedStorage,
    /// The supplied texture is not one ordinary two-dimensional image.
    UnsupportedTextureType,
    /// The texture format does not match the exact requested transfer encoding.
    UnsupportedFormat,
    /// The requested mip level does not exist.
    Level,
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

#[derive(Clone, Copy)]
enum TextureTransferEncoding {
    Rgba8,
    Rgba32Uint,
}

impl TextureTransferEncoding {
    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => 4,
            Self::Rgba32Uint => std::mem::size_of::<[u32; 4]>(),
        }
    }

    fn supports(self, format: MTLPixelFormat) -> bool {
        match self {
            Self::Rgba8 => supports_four_byte_color(format),
            Self::Rgba32Uint => format == MTLPixelFormat::RGBA32Uint,
        }
    }
}

fn checked_texture_region(
    texture: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    origin: [usize; 2],
    extent: [usize; 2],
    encoding: TextureTransferEncoding,
    byte_len: usize,
) -> Result<(MTLRegion, usize), TextureTransferError> {
    if texture.storageMode() != MTLStorageMode::Shared {
        return Err(TextureTransferError::UnsupportedStorage);
    }
    if texture.textureType() != MTLTextureType::Type2D {
        return Err(TextureTransferError::UnsupportedTextureType);
    }
    if !encoding.supports(texture.pixelFormat()) {
        return Err(TextureTransferError::UnsupportedFormat);
    }
    if level >= texture.mipmapLevelCount() {
        return Err(TextureTransferError::Level);
    }
    let shift = u32::try_from(level).map_err(|_| TextureTransferError::Level)?;
    let level_width = texture.width().checked_shr(shift).unwrap_or(0).max(1);
    let level_height = texture.height().checked_shr(shift).unwrap_or(0).max(1);
    let end_x = origin[0]
        .checked_add(extent[0])
        .ok_or(TextureTransferError::Region)?;
    let end_y = origin[1]
        .checked_add(extent[1])
        .ok_or(TextureTransferError::Region)?;
    if extent.contains(&0) || end_x > level_width || end_y > level_height {
        return Err(TextureTransferError::Region);
    }
    let bytes_per_row = extent[0]
        .checked_mul(encoding.bytes_per_pixel())
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
    let (region, bytes_per_row) = checked_texture_region(
        texture,
        level,
        origin,
        extent,
        TextureTransferEncoding::Rgba8,
        bytes.len(),
    )?;
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
    let (region, bytes_per_row) = checked_texture_region(
        texture,
        level,
        origin,
        extent,
        TextureTransferEncoding::Rgba8,
        byte_len,
    )?;
    let mut bytes = vec![0; byte_len];
    let pointer = NonNull::from(bytes.as_mut_slice()).cast::<c_void>();
    // SAFETY: validation above proves the destination supplies the exact writable byte interval
    // Metal fills for this four-byte-per-pixel region and row stride.
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(pointer, bytes_per_row, region, level)
    };
    Ok(bytes)
}

/// Reads one exact two-dimensional four-channel `u32` texture region.
///
/// The result is tightly packed in increasing texture-coordinate row-major order. No format
/// conversion, row padding, or implementation-defined integer normalization is permitted at this
/// boundary.
pub fn read_texture_rgba32_u32(
    texture: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    origin: [usize; 2],
    extent: [usize; 2],
) -> Result<Vec<[u32; 4]>, TextureTransferError> {
    let pixel_count = extent[0]
        .checked_mul(extent[1])
        .ok_or(TextureTransferError::ByteCount)?;
    let byte_len = pixel_count
        .checked_mul(std::mem::size_of::<[u32; 4]>())
        .ok_or(TextureTransferError::ByteCount)?;
    let (region, bytes_per_row) = checked_texture_region(
        texture,
        level,
        origin,
        extent,
        TextureTransferEncoding::Rgba32Uint,
        byte_len,
    )?;
    let mut words = vec![[0_u32; 4]; pixel_count];
    let pointer = NonNull::from(words.as_mut_slice()).cast::<c_void>();
    // SAFETY: the checks above prove that `words` supplies the exact writable byte interval Metal
    // fills for this ordinary shared `RGBA32Uint` region and tightly packed row stride.
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(pointer, bytes_per_row, region, level)
    };
    Ok(words)
}

/// Replaces one exact two-dimensional four-channel `u32` texture region.
pub fn replace_texture_rgba32_u32(
    texture: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    origin: [usize; 2],
    extent: [usize; 2],
    words: &[[u32; 4]],
) -> Result<(), TextureTransferError> {
    let pixel_count = extent[0]
        .checked_mul(extent[1])
        .ok_or(TextureTransferError::ByteCount)?;
    if words.len() != pixel_count {
        return Err(TextureTransferError::ByteCount);
    }
    let byte_len = words
        .len()
        .checked_mul(std::mem::size_of::<[u32; 4]>())
        .ok_or(TextureTransferError::ByteCount)?;
    let (region, bytes_per_row) = checked_texture_region(
        texture,
        level,
        origin,
        extent,
        TextureTransferEncoding::Rgba32Uint,
        byte_len,
    )?;
    let pointer = NonNull::from(words).cast::<c_void>();
    // SAFETY: the checks above prove that `words` supplies the exact readable byte interval Metal
    // consumes for this ordinary shared `RGBA32Uint` region and tightly packed row stride.
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

/// One first-recorded exact drawable result for a fixed layer generation.
///
/// A drawable can be dropped without a presentation callback. This observation separates the
/// evidence lifetime from individual completed render/resource handles: a later callback from a
/// retired attempt can still supply its exact evidence. It retains no drawable or render resource.
///
/// This type makes no image-equivalence claim. A renderer that re-presents one logical image must
/// keep that image immutable and independently check every command's outcome before accepting the
/// observation. The result is the first validated result recorded, not the earliest display time.
pub struct DrawablePresentationObservation {
    generation: DrawablePresentationGeneration,
    result: Arc<ObservedDrawableResult>,
    completed: bool,
}

type ObservedDrawableResult = OnceLock<Result<PresentedDrawable, PresentedDrawableError>>;

impl DrawablePresentationObservation {
    /// Starts an independent observation bound to exactly one immutable layer generation.
    pub fn new(generation: DrawablePresentationGeneration) -> Self {
        Self {
            generation,
            result: Arc::new(OnceLock::new()),
            completed: false,
        }
    }

    /// Consumes the first recorded positive presentation or identity/protocol failure once.
    ///
    /// A per-attempt `NotPresented` result does not complete this observation. No amount of
    /// elapsed time or command completion supplies a positive result. A callback arriving after
    /// another result has won cannot overwrite that result.
    pub fn try_complete(&mut self) -> Result<PresentedDrawableProgress, PresentedDrawableError> {
        if self.completed {
            return Err(PresentedDrawableError::AlreadyCompleted);
        }
        match self.result.get().copied() {
            Some(result) => {
                self.completed = true;
                result.map(PresentedDrawableProgress::Presented)
            }
            None => Ok(PresentedDrawableProgress::Pending),
        }
    }

    fn ensure_live(&self) -> Result<(), PresentedDrawableError> {
        if self.completed {
            Err(PresentedDrawableError::AlreadyCompleted)
        } else {
            Ok(())
        }
    }
}

fn checked_drawable_evidence(
    expected_address: NonNull<()>,
    expected_id: usize,
    actual_address: NonNull<()>,
    actual_id: usize,
    presented_time: f64,
) -> Result<PresentedDrawable, PresentedDrawableError> {
    if actual_address != expected_address || actual_id != expected_id {
        Err(PresentedDrawableError::Identity)
    } else if !presented_time.is_finite() || presented_time <= 0.0 {
        Err(PresentedDrawableError::NotPresented)
    } else {
        Ok(PresentedDrawable {
            drawable_id: actual_id,
            presented_time,
        })
    }
}

fn record_observed_drawable(
    observation: &ObservedDrawableResult,
    result: Result<PresentedDrawable, PresentedDrawableError>,
) {
    if result != Err(PresentedDrawableError::NotPresented) {
        // First validated recorded result wins. The cell is bounded even if every physical
        // attempt is dropped; a duplicate or late callback cannot replace an accepted result.
        let _ = observation.set(result);
    }
}

type PresentedBlock = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLDrawable>>) + 'static>;

/// Owned one-shot completion for one exact registered Metal drawable.
pub struct PresentedDrawableCompletion {
    receiver: mpsc::Receiver<Result<PresentedDrawable, PresentedDrawableError>>,
    _block: PresentedBlock,
    expected_address: NonNull<()>,
    expected_id: usize,
    completed: bool,
    observation: Option<Arc<ObservedDrawableResult>>,
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
                if let Some(observation) = &self.observation {
                    record_observed_drawable(observation, result);
                }
                result.map(PresentedDrawableProgress::Presented)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(PresentedDrawableProgress::Pending),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.completed = true;
                Err(PresentedDrawableError::Cancelled)
            }
        }
    }

    fn try_complete_or_observe(
        &mut self,
        drawable: &ProtocolObject<dyn MTLDrawable>,
    ) -> Result<PresentedDrawableProgress, PresentedDrawableError> {
        let progress = self.try_complete()?;
        if !matches!(progress, PresentedDrawableProgress::Pending) {
            return Ok(progress);
        }
        let actual_address = NonNull::from(drawable).cast::<()>();
        let actual_id = drawable.drawableID();
        let presented_time = drawable.presentedTime();
        self.observe_exact_drawable(actual_address, actual_id, presented_time)
    }

    fn observe_exact_drawable(
        &mut self,
        actual_address: NonNull<()>,
        actual_id: usize,
        presented_time: f64,
    ) -> Result<PresentedDrawableProgress, PresentedDrawableError> {
        let evidence = match checked_drawable_evidence(
            self.expected_address,
            self.expected_id,
            actual_address,
            actual_id,
            presented_time,
        ) {
            Ok(evidence) => evidence,
            Err(PresentedDrawableError::NotPresented) => {
                return Ok(PresentedDrawableProgress::Pending);
            }
            Err(error) => {
                self.completed = true;
                if let Some(observation) = &self.observation {
                    record_observed_drawable(observation, Err(error));
                }
                return Err(error);
            }
        };
        self.completed = true;
        if let Some(observation) = &self.observation {
            record_observed_drawable(observation, Ok(evidence));
        }
        Ok(PresentedDrawableProgress::Presented(evidence))
    }
}

/// Registers one owned post-presentation callback for the exact supplied drawable.
pub fn register_presented_drawable(
    drawable: &ProtocolObject<dyn MTLDrawable>,
) -> PresentedDrawableCompletion {
    register_presented_drawable_with_wake(drawable, None)
}

pub(crate) fn register_presented_drawable_with_wake(
    drawable: &ProtocolObject<dyn MTLDrawable>,
    wake: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
) -> PresentedDrawableCompletion {
    register_presented_drawable_with_observation(drawable, wake, None)
}

fn register_presented_drawable_with_observation(
    drawable: &ProtocolObject<dyn MTLDrawable>,
    wake: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
    observation: Option<Arc<ObservedDrawableResult>>,
) -> PresentedDrawableCompletion {
    let expected_address = NonNull::from(drawable).cast::<()>();
    let expected_id = drawable.drawableID();
    let (sender, receiver) = mpsc::sync_channel(1);
    let callback_observation = observation.clone();
    let block = RcBlock::new(move |actual: NonNull<ProtocolObject<dyn MTLDrawable>>| {
        let actual_address = actual.cast::<()>();
        // SAFETY: Metal invokes `MTLDrawablePresentedHandler` with a valid drawable pointer
        // for the duration of this callback.
        let actual = unsafe { actual.as_ref() };
        let actual_id = actual.drawableID();
        let presented_time = actual.presentedTime();
        let result = checked_drawable_evidence(
            expected_address,
            expected_id,
            actual_address,
            actual_id,
            presented_time,
        );
        if let Some(observation) = &callback_observation {
            record_observed_drawable(observation, result);
        }
        let _ = sender.try_send(result);
        if let Some(wake) = wake.as_ref() {
            // A caller panic must never unwind through Metal's Objective-C callback frame. The
            // completion result remains available to ordinary polling when notification fails.
            let _ = catch_unwind(AssertUnwindSafe(|| wake()));
        }
    });
    // SAFETY: the heap-owned block remains alive in `PresentedDrawableCompletion`; its signature
    // exactly matches `MTLDrawablePresentedHandler`, and Metal copies/retains registered handlers.
    unsafe { drawable.addPresentedHandler(RcBlock::as_ptr(&block)) };
    PresentedDrawableCompletion {
        receiver,
        _block: block,
        expected_address,
        expected_id,
        completed: false,
        observation,
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    use super::*;

    fn observation() -> DrawablePresentationObservation {
        let layer = CAMetalLayer::new();
        let mut extent = layer.drawableSize();
        extent.width = 96.0;
        extent.height = 96.0;
        layer.setDrawableSize(extent);
        DrawablePresentationObservation::new(
            DrawablePresentationGeneration::try_new(layer, 96, 96, 1).unwrap(),
        )
    }

    fn observed(id: usize, time: f64) -> Result<PresentedDrawable, PresentedDrawableError> {
        let identity = ();
        let address = NonNull::from(&identity);
        checked_drawable_evidence(address, id, address, id, time)
    }

    #[test]
    fn retired_attempt_callback_outlives_repeated_physical_replacement() {
        let mut observation = observation();
        // This Arc models the callback copied/retained by Metal, not a drawable resource owner.
        let delayed_callback = Arc::clone(&observation.result);
        for _service_opportunity in 0..10_000 {
            let attempt = Arc::clone(&observation.result);
            record_observed_drawable(&attempt, Err(PresentedDrawableError::NotPresented));
            drop(attempt);
            assert_eq!(
                observation.try_complete(),
                Ok(PresentedDrawableProgress::Pending)
            );
        }
        assert!(observation.result.get().is_none());
        record_observed_drawable(&delayed_callback, observed(17, 1.5));
        assert_eq!(
            observation.try_complete(),
            Ok(PresentedDrawableProgress::Presented(PresentedDrawable {
                drawable_id: 17,
                presented_time: 1.5
            })),
        );
        assert_eq!(
            observation.try_complete(),
            Err(PresentedDrawableError::AlreadyCompleted)
        );
        assert_eq!(
            observation.ensure_live(),
            Err(PresentedDrawableError::AlreadyCompleted)
        );
    }

    #[test]
    fn first_valid_recording_wins_even_if_later_callback_has_earlier_time() {
        let mut observation = observation();
        let first = observed(9, 3.0);
        // The exact getter and callback both use this identical bounded recording operation.
        record_observed_drawable(&observation.result, first);
        record_observed_drawable(&observation.result, observed(2, 1.0));
        record_observed_drawable(&observation.result, Err(PresentedDrawableError::Identity));
        assert_eq!(
            observation.try_complete(),
            first.map(PresentedDrawableProgress::Presented),
        );
    }

    #[test]
    fn exact_getter_before_callback_populates_the_same_once_observation() {
        let mut observation = observation();
        let identity = ();
        let address = NonNull::from(&identity);
        let (_sender, receiver) = mpsc::sync_channel(1);
        let block: PresentedBlock = RcBlock::new(|_| {});
        let mut attempt = PresentedDrawableCompletion {
            receiver,
            _block: block,
            expected_address: address,
            expected_id: 7,
            completed: false,
            observation: Some(Arc::clone(&observation.result)),
        };
        assert_eq!(
            attempt.try_complete(),
            Ok(PresentedDrawableProgress::Pending)
        );
        assert_eq!(
            attempt.observe_exact_drawable(address, 7, 0.0),
            Ok(PresentedDrawableProgress::Pending),
        );
        assert_eq!(
            observation.try_complete(),
            Ok(PresentedDrawableProgress::Pending)
        );
        let evidence = PresentedDrawable {
            drawable_id: 7,
            presented_time: 2.0,
        };
        assert_eq!(
            attempt.observe_exact_drawable(address, 7, 2.0),
            Ok(PresentedDrawableProgress::Presented(evidence)),
        );
        drop(attempt);
        assert_eq!(
            observation.try_complete(),
            Ok(PresentedDrawableProgress::Presented(evidence))
        );
        // The old callback can arrive later without replacing the winning evidence or reopening
        // enrollment. No physical resource needs to remain owned for this late result.
        record_observed_drawable(&observation.result, observed(7, 2.0));
        assert_eq!(
            observation.ensure_live(),
            Err(PresentedDrawableError::AlreadyCompleted)
        );
        assert_eq!(
            observation.try_complete(),
            Err(PresentedDrawableError::AlreadyCompleted)
        );
    }

    #[test]
    fn exact_identity_failure_refuses_and_dropped_attempt_is_not_success() {
        let mut observation = observation();
        let identities = [0_u8, 1_u8];
        let a = NonNull::from(&identities[0]).cast();
        let b = NonNull::from(&identities[1]).cast();
        for time in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(observed(9, time), Err(PresentedDrawableError::NotPresented));
        }
        for result in [
            checked_drawable_evidence(a, 9, b, 9, 1.0),
            checked_drawable_evidence(a, 9, a, 10, 1.0),
        ] {
            assert_eq!(result, Err(PresentedDrawableError::Identity));
            record_observed_drawable(&observation.result, result);
        }
        record_observed_drawable(&observation.result, observed(9, 1.0));
        assert_eq!(
            observation.try_complete(),
            Err(PresentedDrawableError::Identity)
        );
    }

    #[test]
    fn registered_callback_records_exact_failure_after_completion_owner_is_dropped() {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("test host has Metal");
        let mut observation = observation();
        observation.generation.layer.setDevice(Some(&device));
        let expected = observation
            .generation
            .layer
            .nextDrawable()
            .expect("first drawable");
        let foreign = observation
            .generation
            .layer
            .nextDrawable()
            .expect("distinct drawable");
        let expected_drawable: &ProtocolObject<dyn CAMetalDrawable> = &expected;
        let completion = register_presented_drawable_with_observation(
            expected_drawable.as_ref(),
            None,
            Some(Arc::clone(&observation.result)),
        );
        // Invoke the actual registered block after retiring its completion owner, with a genuine
        // but foreign drawable. This tests callback wiring/lifetime, not physical display success.
        let callback = completion._block.clone();
        drop(completion);
        drop(expected);
        let foreign_drawable: &ProtocolObject<dyn CAMetalDrawable> = &foreign;
        callback.call((NonNull::from(foreign_drawable.as_ref()),));
        assert_eq!(
            observation.try_complete(),
            Err(PresentedDrawableError::Identity)
        );
    }

    #[test]
    fn external_epoch_keeps_returned_extent_distinct() {
        let layer = CAMetalLayer::new();
        let mut extent = layer.drawableSize();
        extent.width = 96.0;
        extent.height = 96.0;
        layer.setDrawableSize(extent);
        let first = DrawablePresentationGeneration::try_new(layer.clone(), 96, 96, 1)
            .expect("first external epoch is admitted");
        let returned = DrawablePresentationGeneration::try_new(layer, 96, 96, 3)
            .expect("returned extent carries its later external epoch");
        assert_ne!(first, returned);
    }

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

        let private = device
            .newBufferWithLength_options(initial.len(), MTLResourceOptions::StorageModePrivate)
            .expect("private allocation works");
        assert_eq!(
            write_shared_buffer(&private, 0, &initial),
            Err(BufferTransferError::UnsupportedStorage)
        );
        assert_eq!(
            read_shared_buffer(&private, 0..initial.len()),
            Err(BufferTransferError::UnsupportedStorage)
        );
    }

    #[test]
    fn metal_shared_rgba32_u32_round_trip_is_exact_and_shape_checked() {
        let device = MTLCreateSystemDefaultDevice().expect("test host has a Metal device");
        let texture = new_texture_2d(
            &device,
            2,
            2,
            MTLPixelFormat::RGBA32Uint,
            Texture2DStorage::Shared,
            Texture2DUse::RenderTarget,
            Texture2DMips::One,
        )
        .expect("integer texture allocation works");
        let words = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [u32::MAX, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        replace_texture_rgba32_u32(&texture, 0, [0, 0], [2, 2], &words)
            .expect("integer replacement works");
        assert_eq!(
            read_texture_rgba32_u32(&texture, 0, [0, 0], [2, 2]).expect("integer readback works"),
            words
        );
        assert_eq!(
            replace_texture_rgba32_u32(&texture, 0, [0, 0], [2, 2], &words[..3]),
            Err(TextureTransferError::ByteCount)
        );
    }
}
