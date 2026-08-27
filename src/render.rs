//! Checked ownership around one exact Metal render submission.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::{NonNull, addr_eq};

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCullMode, MTLDepthStencilState, MTLDevice, MTLDrawable, MTLLoadAction,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor as RawRenderPassDescriptor, MTLRenderPipelineColorAttachmentDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLSamplerState, MTLScissorRect,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage, MTLViewport, MTLWinding,
};
use objc2_quartz_core::CAMetalDrawable;

use crate::{
    PresentedDrawableCompletion, PresentedDrawableError, PresentedDrawableProgress,
    register_presented_drawable,
};

const COLOR_ATTACHMENT_COUNT: usize = 8;
const BUFFER_ARGUMENT_COUNT: usize = 31;
const TEXTURE_ARGUMENT_COUNT: usize = 128;
const SAMPLER_ARGUMENT_COUNT: usize = 16;
const MAX_INLINE_BYTES: usize = 4_096;
const MAX_2D_TEXTURE_DIMENSION: usize = 16_384;

/// CPU visibility and residency requested for one checked two-dimensional texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Texture2DStorage {
    /// CPU-visible coherent storage, used for deterministic upload/readback targets.
    Shared,
    /// GPU-private storage, used for ordinary render and sampled targets.
    Private,
}

/// Exact intended use of one checked two-dimensional texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Texture2DUse {
    /// Render attachment only.
    RenderTarget,
    /// Shader sampling only.
    Sampled,
    /// Render attachment followed by shader sampling.
    RenderTargetAndSampled,
}

/// Mipmap allocation requested for one checked two-dimensional texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Texture2DMips {
    /// Allocate only level zero.
    One,
    /// Allocate the complete dimension-derived mip chain down to one texel.
    Complete,
}

/// Why one checked two-dimensional Metal texture could not be allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureAllocationError {
    /// Width or height was zero or exceeded the safe cross-device dimension bound.
    Dimensions,
    /// The pixel format is outside this boundary's explicit render/sample formats.
    PixelFormat,
    /// Metal refused the validated texture allocation.
    Allocation,
}

/// Allocates one ordinary non-mipmapped two-dimensional texture from checked exact properties.
pub fn new_texture_2d(
    device: &ProtocolObject<dyn MTLDevice>,
    width: usize,
    height: usize,
    pixel_format: MTLPixelFormat,
    storage: Texture2DStorage,
    usage: Texture2DUse,
    mipmaps: Texture2DMips,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, TextureAllocationError> {
    if width == 0
        || height == 0
        || width > MAX_2D_TEXTURE_DIMENSION
        || height > MAX_2D_TEXTURE_DIMENSION
    {
        return Err(TextureAllocationError::Dimensions);
    }
    if !matches!(
        pixel_format,
        MTLPixelFormat::R8Unorm
            | MTLPixelFormat::RG8Unorm
            | MTLPixelFormat::RGBA8Unorm
            | MTLPixelFormat::RGBA8Unorm_sRGB
            | MTLPixelFormat::BGRA8Unorm
            | MTLPixelFormat::BGRA8Unorm_sRGB
            | MTLPixelFormat::RGBA32Uint
            | MTLPixelFormat::RGBA16Float
            | MTLPixelFormat::Depth32Float
    ) {
        return Err(TextureAllocationError::PixelFormat);
    }
    // SAFETY: both dimensions are nonzero and bounded to Metal's maximum modern 2D dimension;
    // the unsupported-device case is reported by the fallible allocation below.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            pixel_format,
            width,
            height,
            mipmaps == Texture2DMips::Complete,
        )
    };
    descriptor.setStorageMode(match storage {
        Texture2DStorage::Shared => MTLStorageMode::Shared,
        Texture2DStorage::Private => MTLStorageMode::Private,
    });
    descriptor.setUsage(match usage {
        Texture2DUse::RenderTarget => MTLTextureUsage::RenderTarget,
        Texture2DUse::Sampled => MTLTextureUsage::ShaderRead,
        Texture2DUse::RenderTargetAndSampled => {
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead
        }
    });
    device
        .newTextureWithDescriptor(&descriptor)
        .ok_or(TextureAllocationError::Allocation)
}

/// Why checked Metal render encoding could not proceed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderCommandError {
    /// Metal did not create a command buffer for the supplied queue.
    CommandBufferAllocation,
    /// Metal did not create a render encoder for the supplied attachments.
    EncoderAllocation,
    /// A color attachment index exceeded Metal's fixed render-target count.
    ColorAttachmentIndex,
    /// A color or depth attachment was not one ordinary two-dimensional texture.
    AttachmentTextureType,
    /// A color or depth attachment used a pixel format belonging to the other attachment class.
    AttachmentPixelFormat,
    /// A clear value was non-finite, or a depth clear was outside the exact normalized interval.
    ClearValue,
    /// A shader buffer argument index exceeded Metal's fixed argument count.
    BufferArgumentIndex,
    /// A shader texture argument index exceeded Metal's fixed argument count.
    TextureArgumentIndex,
    /// A shader sampler argument index exceeded Metal's fixed argument count.
    SamplerArgumentIndex,
    /// An inline byte binding was empty or exceeded Metal's exact inline-byte limit.
    InlineByteCount,
    /// A requested byte offset did not fit the exact supplied Metal buffer.
    BufferRange,
    /// Vertex records had zero stride/count or did not fit the exact supplied buffer.
    VertexRecords,
    /// A draw referred to another pass, a superseded binding, or vertices outside its records.
    DrawRange,
    /// The supplied drawable texture was not rendered by this exact command buffer.
    DrawableTarget,
    /// Metal completed the command buffer in a non-success state.
    Execution(MTLCommandBufferStatus),
    /// Metal did not report valid terminal evidence for the exact drawable.
    Presentation(PresentedDrawableError),
}

/// One explicit color-load operation for a render attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorLoad {
    /// Preserve the attachment's current pixels.
    Load,
    /// Discard current pixels before rendering.
    DontCare,
    /// Clear every pixel to this exact linear RGBA value.
    Clear([f64; 4]),
}

/// One explicit color-store operation for a render attachment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorStore {
    /// Preserve rendered pixels after the pass.
    Store,
    /// Permit Metal to discard rendered pixels after the pass.
    DontCare,
}

/// An owned render-pass description whose attachment indices and resource lifetimes are checked.
pub struct RenderPassDescriptor {
    raw: Retained<RawRenderPassDescriptor>,
    color_targets: [Option<Retained<ProtocolObject<dyn MTLTexture>>>; COLOR_ATTACHMENT_COUNT],
    depth_target: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

impl RenderPassDescriptor {
    /// Creates an empty pass with no implicit attachment or load/store policy.
    pub fn new() -> Self {
        Self {
            raw: RawRenderPassDescriptor::renderPassDescriptor(),
            color_targets: std::array::from_fn(|_| None),
            depth_target: None,
        }
    }

    /// Binds one exact two-dimensional color target at a checked Metal attachment index.
    pub fn set_color_attachment(
        &mut self,
        index: usize,
        texture: &ProtocolObject<dyn MTLTexture>,
        load: ColorLoad,
        store: ColorStore,
    ) -> Result<(), RenderCommandError> {
        if index >= COLOR_ATTACHMENT_COUNT {
            return Err(RenderCommandError::ColorAttachmentIndex);
        }
        if texture.textureType() != MTLTextureType::Type2D {
            return Err(RenderCommandError::AttachmentTextureType);
        }
        if texture.pixelFormat() == MTLPixelFormat::Depth32Float {
            return Err(RenderCommandError::AttachmentPixelFormat);
        }
        if let ColorLoad::Clear(rgba) = load {
            if !rgba.iter().all(|channel| channel.is_finite()) {
                return Err(RenderCommandError::ClearValue);
            }
        }
        let attachments = self.raw.colorAttachments();
        // SAFETY: the index was checked against Metal's fixed eight-element attachment array.
        let attachment = unsafe { attachments.objectAtIndexedSubscript(index) };
        attachment.setTexture(Some(texture));
        match load {
            ColorLoad::Load => attachment.setLoadAction(MTLLoadAction::Load),
            ColorLoad::DontCare => attachment.setLoadAction(MTLLoadAction::DontCare),
            ColorLoad::Clear(rgba) => {
                attachment.setLoadAction(MTLLoadAction::Clear);
                attachment.setClearColor(MTLClearColor {
                    red: rgba[0],
                    green: rgba[1],
                    blue: rgba[2],
                    alpha: rgba[3],
                });
            }
        }
        attachment.setStoreAction(match store {
            ColorStore::Store => MTLStoreAction::Store,
            ColorStore::DontCare => MTLStoreAction::DontCare,
        });
        self.color_targets[index] = Some(texture.retain());
        Ok(())
    }

    /// Binds one exact two-dimensional depth target and its clear/store policy.
    pub fn set_depth_attachment(
        &mut self,
        texture: &ProtocolObject<dyn MTLTexture>,
        clear_depth: Option<f64>,
        store: bool,
    ) -> Result<(), RenderCommandError> {
        if texture.textureType() != MTLTextureType::Type2D {
            return Err(RenderCommandError::AttachmentTextureType);
        }
        if texture.pixelFormat() != MTLPixelFormat::Depth32Float {
            return Err(RenderCommandError::AttachmentPixelFormat);
        }
        if let Some(clear_depth) = clear_depth {
            if !clear_depth.is_finite() || !(0.0..=1.0).contains(&clear_depth) {
                return Err(RenderCommandError::ClearValue);
            }
        }
        let attachment = self.raw.depthAttachment();
        attachment.setTexture(Some(texture));
        if let Some(clear_depth) = clear_depth {
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setClearDepth(clear_depth);
        } else {
            attachment.setLoadAction(MTLLoadAction::Load);
        }
        attachment.setStoreAction(if store {
            MTLStoreAction::Store
        } else {
            MTLStoreAction::DontCare
        });
        self.depth_target = Some(texture.retain());
        Ok(())
    }
}

impl Default for RenderPassDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a checked render-pipeline attachment lookup could not proceed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderPipelineAttachmentError {
    /// The index exceeded Metal's fixed render-target count.
    Index,
}

/// Obtains one render-pipeline color attachment after checking Metal's fixed array bound.
pub fn render_pipeline_color_attachment(
    descriptor: &MTLRenderPipelineDescriptor,
    index: usize,
) -> Result<Retained<MTLRenderPipelineColorAttachmentDescriptor>, RenderPipelineAttachmentError> {
    if index >= COLOR_ATTACHMENT_COUNT {
        return Err(RenderPipelineAttachmentError::Index);
    }
    let attachments = descriptor.colorAttachments();
    // SAFETY: the index was checked against Metal's fixed eight-element attachment array.
    Ok(unsafe { attachments.objectAtIndexedSubscript(index) })
}

/// Exact records proved to fit the vertex buffer currently bound in one render pass.
pub struct BoundVertexRecords<'command> {
    encoder: NonNull<()>,
    slot: usize,
    binding: VertexBinding,
    _command: PhantomData<&'command mut RenderCommandBuffer>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VertexBinding {
    buffer: NonNull<()>,
    offset: usize,
    stride: usize,
    count: usize,
}

/// One retained Metal command buffer that may contain one or more sequential render passes.
pub struct RenderCommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    rendered_color_targets: Vec<NonNull<()>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    depth_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
}

impl RenderCommandBuffer {
    /// Creates one retained-resource command buffer from the exact supplied Metal queue.
    pub fn new(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<Self, RenderCommandError> {
        let raw = queue
            .commandBuffer()
            .ok_or(RenderCommandError::CommandBufferAllocation)?;
        Ok(Self {
            raw,
            rendered_color_targets: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            pipelines: Vec::new(),
            depth_states: Vec::new(),
        })
    }

    /// Begins one pass while exclusively borrowing this command buffer until the pass ends.
    pub fn begin_render_pass<'command>(
        &'command mut self,
        descriptor: &RenderPassDescriptor,
    ) -> Result<RenderPass<'command>, RenderCommandError> {
        let encoder = self
            .raw
            .renderCommandEncoderWithDescriptor(&descriptor.raw)
            .ok_or(RenderCommandError::EncoderAllocation)?;
        for texture in descriptor.color_targets.iter().flatten() {
            let texture_ref: &ProtocolObject<dyn MTLTexture> = texture;
            self.rendered_color_targets
                .push(NonNull::from(texture_ref).cast::<()>());
            self.textures.push(texture.clone());
        }
        if let Some(texture) = &descriptor.depth_target {
            self.textures.push(texture.clone());
        }
        Ok(RenderPass {
            owner: self,
            encoder,
            vertex_bindings: [None; BUFFER_ARGUMENT_COUNT],
            ended: false,
        })
    }

    /// Commits and synchronously verifies this exact offscreen command buffer.
    pub fn commit_and_wait(self) -> Result<(), RenderCommandError> {
        self.raw.commit();
        self.raw.waitUntilCompleted();
        let status = self.raw.status();
        if status == MTLCommandBufferStatus::Completed {
            Ok(())
        } else {
            Err(RenderCommandError::Execution(status))
        }
    }

    /// Commits this command buffer and presents the exact drawable it rendered.
    pub fn present(
        self,
        drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    ) -> Result<PendingPresentedRender, RenderCommandError> {
        let drawable_texture = drawable.texture();
        let drawable_texture_ref: &ProtocolObject<dyn MTLTexture> = &drawable_texture;
        let drawable_address = NonNull::from(drawable_texture_ref).cast::<()>();
        if !self
            .rendered_color_targets
            .iter()
            .any(|target| addr_eq(target.as_ptr(), drawable_address.as_ptr()))
        {
            return Err(RenderCommandError::DrawableTarget);
        }
        let camera_drawable: &ProtocolObject<dyn CAMetalDrawable> = &drawable;
        let metal_drawable: &ProtocolObject<dyn MTLDrawable> = camera_drawable.as_ref();
        let completion = register_presented_drawable(metal_drawable);
        self.raw.presentDrawable(metal_drawable);
        self.raw.commit();
        Ok(PendingPresentedRender {
            completion,
            command_buffer: self.raw,
            _drawable: drawable,
            _buffers: self.buffers,
            _textures: self.textures,
            _samplers: self.samplers,
            _pipelines: self.pipelines,
            _depth_states: self.depth_states,
        })
    }
}

/// One active Metal render pass; dropping it ends encoding exactly once.
pub struct RenderPass<'command> {
    owner: &'command mut RenderCommandBuffer,
    encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    vertex_bindings: [Option<VertexBinding>; BUFFER_ARGUMENT_COUNT],
    ended: bool,
}

impl<'command> RenderPass<'command> {
    /// Ends this pass, releasing its exclusive borrow of the command buffer.
    pub fn end(mut self) {
        self.end_once();
    }

    /// Selects one exact compiled render pipeline for subsequent draws.
    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLRenderPipelineState>) {
        self.encoder.setRenderPipelineState(pipeline);
        self.owner.pipelines.push(pipeline.retain());
    }

    /// Selects one exact compiled depth/stencil state for subsequent draws.
    pub fn set_depth_stencil(&mut self, state: &ProtocolObject<dyn MTLDepthStencilState>) {
        self.encoder.setDepthStencilState(Some(state));
        self.owner.depth_states.push(state.retain());
    }

    /// Sets one viewport without exposing the raw render encoder.
    pub fn set_viewport(&mut self, viewport: MTLViewport) {
        self.encoder.setViewport(viewport);
    }

    /// Sets one scissor rectangle without exposing the raw render encoder.
    pub fn set_scissor(&mut self, scissor: MTLScissorRect) {
        self.encoder.setScissorRect(scissor);
    }

    /// Sets triangle culling without exposing the raw render encoder.
    pub fn set_cull_mode(&mut self, mode: MTLCullMode) {
        self.encoder.setCullMode(mode);
    }

    /// Sets front-face winding without exposing the raw render encoder.
    pub fn set_front_facing_winding(&mut self, winding: MTLWinding) {
        self.encoder.setFrontFacingWinding(winding);
    }

    /// Copies exact initialized bytes into one checked vertex-stage argument slot.
    pub fn set_vertex_bytes(
        &mut self,
        index: usize,
        bytes: &[u8],
    ) -> Result<(), RenderCommandError> {
        check_buffer_argument(index)?;
        check_inline_bytes(bytes)?;
        self.vertex_bindings[index] = None;
        let pointer = NonNull::from(bytes).cast::<c_void>();
        // SAFETY: `bytes` supplies exactly the initialized copied interval, its nonzero length is
        // bounded by Metal's inline limit, and the argument index was checked above.
        unsafe {
            self.encoder
                .setVertexBytes_length_atIndex(pointer, bytes.len(), index)
        };
        Ok(())
    }

    /// Copies exact initialized bytes into one checked fragment-stage argument slot.
    pub fn set_fragment_bytes(
        &mut self,
        index: usize,
        bytes: &[u8],
    ) -> Result<(), RenderCommandError> {
        check_buffer_argument(index)?;
        check_inline_bytes(bytes)?;
        let pointer = NonNull::from(bytes).cast::<c_void>();
        // SAFETY: `bytes` supplies exactly the initialized copied interval, its nonzero length is
        // bounded by Metal's inline limit, and the argument index was checked above.
        unsafe {
            self.encoder
                .setFragmentBytes_length_atIndex(pointer, bytes.len(), index)
        };
        Ok(())
    }

    /// Binds one fragment-stage buffer at a checked byte offset and retains it through completion.
    pub fn set_fragment_buffer(
        &mut self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) -> Result<(), RenderCommandError> {
        check_buffer_argument(index)?;
        if offset >= buffer.length() {
            return Err(RenderCommandError::BufferRange);
        }
        // SAFETY: the buffer is retained by `owner`, its starting offset and argument index were
        // checked, and the retaining command buffer synchronizes its use. Shader layout and access
        // remain the caller-owned pipeline contract, as in Metal itself.
        unsafe {
            self.encoder
                .setFragmentBuffer_offset_atIndex(Some(buffer), offset, index)
        };
        self.owner.buffers.push(buffer.retain());
        Ok(())
    }

    /// Binds one checked fragment-stage texture and retains it through command completion.
    pub fn set_fragment_texture(
        &mut self,
        index: usize,
        texture: &ProtocolObject<dyn MTLTexture>,
    ) -> Result<(), RenderCommandError> {
        if index >= TEXTURE_ARGUMENT_COUNT {
            return Err(RenderCommandError::TextureArgumentIndex);
        }
        // SAFETY: the texture is retained by `owner`, the retaining command buffer synchronizes
        // its use, and the argument index was checked against Metal's fixed limit.
        unsafe {
            self.encoder
                .setFragmentTexture_atIndex(Some(texture), index)
        };
        self.owner.textures.push(texture.retain());
        Ok(())
    }

    /// Binds one checked fragment-stage sampler and retains it through command completion.
    pub fn set_fragment_sampler(
        &mut self,
        index: usize,
        sampler: &ProtocolObject<dyn MTLSamplerState>,
    ) -> Result<(), RenderCommandError> {
        if index >= SAMPLER_ARGUMENT_COUNT {
            return Err(RenderCommandError::SamplerArgumentIndex);
        }
        // SAFETY: the sampler is retained by `owner`, and the argument index was checked against
        // Metal's fixed limit.
        unsafe {
            self.encoder
                .setFragmentSamplerState_atIndex(Some(sampler), index)
        };
        self.owner.samplers.push(sampler.retain());
        Ok(())
    }

    /// Binds an exact vertex-record interval and returns proof of its checked record capacity.
    pub fn bind_vertex_records(
        &mut self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Result<BoundVertexRecords<'command>, RenderCommandError> {
        check_buffer_argument(index)?;
        if stride == 0 || count == 0 {
            return Err(RenderCommandError::VertexRecords);
        }
        let byte_count = stride
            .checked_mul(count)
            .ok_or(RenderCommandError::VertexRecords)?;
        let end = offset
            .checked_add(byte_count)
            .ok_or(RenderCommandError::VertexRecords)?;
        if end > buffer.length() {
            return Err(RenderCommandError::VertexRecords);
        }
        let buffer_ref: &ProtocolObject<dyn MTLBuffer> = buffer;
        let binding = VertexBinding {
            buffer: NonNull::from(buffer_ref).cast::<()>(),
            offset,
            stride,
            count,
        };
        self.vertex_bindings[index] = Some(binding);
        // SAFETY: the buffer is retained by `owner`, the complete record interval and argument
        // index were checked, and the retaining command buffer synchronizes resource use.
        unsafe {
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(buffer), offset, index)
        };
        self.owner.buffers.push(buffer.retain());
        let encoder_ref: &ProtocolObject<dyn MTLRenderCommandEncoder> = &self.encoder;
        Ok(BoundVertexRecords {
            encoder: NonNull::from(encoder_ref).cast::<()>(),
            slot: index,
            binding,
            _command: PhantomData,
        })
    }

    /// Draws one checked triangle-list interval from the exact current vertex-record binding.
    pub fn draw_triangles(
        &mut self,
        records: &BoundVertexRecords<'command>,
        vertices: Range<usize>,
    ) -> Result<(), RenderCommandError> {
        let encoder_ref: &ProtocolObject<dyn MTLRenderCommandEncoder> = &self.encoder;
        let encoder = NonNull::from(encoder_ref).cast::<()>();
        let valid_binding = addr_eq(records.encoder.as_ptr(), encoder.as_ptr())
            && records.slot < BUFFER_ARGUMENT_COUNT
            && self.vertex_bindings[records.slot] == Some(records.binding);
        let valid_range = vertices.start < vertices.end
            && vertices.end <= records.binding.count
            && vertices.len() % 3 == 0;
        if !valid_binding || !valid_range {
            return Err(RenderCommandError::DrawRange);
        }
        // SAFETY: the opaque records prove this exact pass still has a live, retained vertex
        // buffer with at least `records.count` complete records, and the triangle range was checked.
        unsafe {
            self.encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                vertices.start,
                vertices.len(),
            )
        };
        Ok(())
    }

    fn end_once(&mut self) {
        if !self.ended {
            self.encoder.endEncoding();
            self.ended = true;
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.end_once();
    }
}

/// One committed render submission retaining its exact drawable and completion evidence.
#[must_use = "a presented render must be polled to terminal drawable evidence"]
pub struct PendingPresentedRender {
    completion: PresentedDrawableCompletion,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
    _buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    _textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    _samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    _pipelines: Vec<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    _depth_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
}

impl PendingPresentedRender {
    /// Polls without blocking for exact drawable presentation and command-buffer success.
    pub fn try_complete(&mut self) -> Result<PresentedDrawableProgress, RenderCommandError> {
        let progress = self
            .completion
            .try_complete()
            .map_err(RenderCommandError::Presentation)?;
        if matches!(progress, PresentedDrawableProgress::Presented(_)) {
            let status = self.command_buffer.status();
            if status == MTLCommandBufferStatus::Error {
                return Err(RenderCommandError::Execution(status));
            }
        }
        Ok(progress)
    }
}

fn check_buffer_argument(index: usize) -> Result<(), RenderCommandError> {
    if index >= BUFFER_ARGUMENT_COUNT {
        Err(RenderCommandError::BufferArgumentIndex)
    } else {
        Ok(())
    }
}

fn check_inline_bytes(bytes: &[u8]) -> Result<(), RenderCommandError> {
    if bytes.is_empty() || bytes.len() > MAX_INLINE_BYTES {
        Err(RenderCommandError::InlineByteCount)
    } else {
        Ok(())
    }
}
