//! Real-device proof for checked render encoding, bounds, and offscreen pixels.

use lbu_metal_safe::{
    ColorLoad, ColorStore, RenderCommandBuffer, RenderCommandError, RenderPassDescriptor,
    RenderPipelineAttachmentError, Texture2DMips, Texture2DStorage, Texture2DUse,
    TextureAllocationError, new_texture_2d, read_texture_rgba8, render_pipeline_color_attachment,
    shared_buffer_with_bytes,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLPixelFormat,
    MTLRenderPipelineDescriptor,
};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut vertex_main(
    const device packed_float2* positions [[buffer(0)]],
    uint vertex_index [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2(positions[vertex_index]), 0.0, 1.0);
    return out;
}

fragment float4 fragment_main(
    constant float4& color [[buffer(0)]])
{
    return color;
}
"#;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[test]
fn metal_render_transaction_draws_exact_checked_triangle() {
    let device = MTLCreateSystemDefaultDevice().expect("test host has a Metal device");
    let queue = device
        .newCommandQueue()
        .expect("command queue allocation works");
    let source = NSString::from_str(SHADER);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("test shader compiles");
    let vertex_name = NSString::from_str("vertex_main");
    let fragment_name = NSString::from_str("fragment_main");
    let vertex = library
        .newFunctionWithName(&vertex_name)
        .expect("vertex function exists");
    let fragment = library
        .newFunctionWithName(&fragment_name)
        .expect("fragment function exists");

    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex));
    pipeline_descriptor.setFragmentFunction(Some(&fragment));
    let pipeline_attachment = render_pipeline_color_attachment(&pipeline_descriptor, 0)
        .expect("attachment zero is bounded");
    pipeline_attachment.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    assert_eq!(
        render_pipeline_color_attachment(&pipeline_descriptor, 8),
        Err(RenderPipelineAttachmentError::Index)
    );
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .expect("render pipeline compiles");

    assert_eq!(
        new_texture_2d(
            &device,
            0,
            32,
            MTLPixelFormat::RGBA8Unorm,
            Texture2DStorage::Shared,
            Texture2DUse::RenderTarget,
            Texture2DMips::One,
        ),
        Err(TextureAllocationError::Dimensions)
    );
    let target = new_texture_2d(
        &device,
        32,
        32,
        MTLPixelFormat::RGBA8Unorm,
        Texture2DStorage::Shared,
        Texture2DUse::RenderTarget,
        Texture2DMips::One,
    )
    .expect("shared render target allocation works");

    let vertices = f32_bytes(&[-0.8, -0.8, 0.8, -0.8, 0.0, 0.8]);
    let vertex_buffer = shared_buffer_with_bytes(&device, &vertices)
        .expect("shared vertex buffer allocation works");

    let mut pass_descriptor = RenderPassDescriptor::new();
    let depth = new_texture_2d(
        &device,
        32,
        32,
        MTLPixelFormat::Depth32Float,
        Texture2DStorage::Private,
        Texture2DUse::RenderTarget,
        Texture2DMips::One,
    )
    .expect("private depth target allocation works");
    assert_eq!(
        pass_descriptor.set_color_attachment(
            8,
            &target,
            ColorLoad::Clear([0.0, 0.0, 0.0, 1.0]),
            ColorStore::Store,
        ),
        Err(RenderCommandError::ColorAttachmentIndex)
    );
    assert_eq!(
        pass_descriptor.set_color_attachment(0, &depth, ColorLoad::DontCare, ColorStore::DontCare,),
        Err(RenderCommandError::AttachmentPixelFormat)
    );
    assert_eq!(
        pass_descriptor.set_color_attachment(
            0,
            &target,
            ColorLoad::Clear([f64::NAN, 0.0, 0.0, 1.0]),
            ColorStore::Store,
        ),
        Err(RenderCommandError::ClearValue)
    );
    assert_eq!(
        pass_descriptor.set_depth_attachment(&target, Some(1.0), false),
        Err(RenderCommandError::AttachmentPixelFormat)
    );
    assert_eq!(
        pass_descriptor.set_depth_attachment(&depth, Some(2.0), false),
        Err(RenderCommandError::ClearValue)
    );
    pass_descriptor
        .set_color_attachment(
            0,
            &target,
            ColorLoad::Clear([0.0, 0.0, 0.0, 1.0]),
            ColorStore::Store,
        )
        .expect("color attachment zero is valid");

    let mut command = RenderCommandBuffer::new(&queue).expect("command buffer allocation works");
    {
        let mut pass = command
            .begin_render_pass(&pass_descriptor)
            .expect("render encoder allocation works");
        pass.set_pipeline(&pipeline);
        assert_eq!(
            pass.set_vertex_bytes(31, &[1]),
            Err(RenderCommandError::BufferArgumentIndex)
        );
        assert_eq!(
            pass.set_fragment_bytes(0, &vec![0; 4_097]),
            Err(RenderCommandError::InlineByteCount)
        );
        assert_eq!(
            pass.bind_vertex_records(0, &vertex_buffer, 0, 8, 4).err(),
            Some(RenderCommandError::VertexRecords)
        );
        let superseded = pass
            .bind_vertex_records(0, &vertex_buffer, 0, 8, 3)
            .expect("three complete positions fit");
        pass.set_vertex_bytes(0, &[0; 8])
            .expect("bounded inline bytes work");
        assert_eq!(
            pass.draw_triangles(&superseded, 0..3),
            Err(RenderCommandError::DrawRange)
        );
        let records = pass
            .bind_vertex_records(0, &vertex_buffer, 0, 8, 3)
            .expect("rebinding mints current record proof");
        assert_eq!(
            pass.draw_triangles(&records, 0..2),
            Err(RenderCommandError::DrawRange)
        );
        pass.set_fragment_bytes(0, &f32_bytes(&[1.0, 0.0, 0.0, 1.0]))
            .expect("exact color bytes fit");
        pass.draw_triangles(&records, 0..3)
            .expect("exact triangle draw is accepted");
        pass.end();
    }
    command
        .commit_and_wait()
        .expect("offscreen render completes successfully");

    let pixels =
        read_texture_rgba8(&target, 0, [0, 0], [32, 32]).expect("shared render target is readable");
    let center = (16 * 32 + 16) * 4;
    assert!(pixels[center] >= 250, "center red={}", pixels[center]);
    assert!(
        pixels[center + 1] <= 5,
        "center green={}",
        pixels[center + 1]
    );
    assert!(
        pixels[center + 2] <= 5,
        "center blue={}",
        pixels[center + 2]
    );
    assert!(
        pixels[center + 3] >= 250,
        "center alpha={}",
        pixels[center + 3]
    );
    assert_eq!(&pixels[0..4], &[0, 0, 0, 255]);
}
