//! Dynamic rendering commands.
//!
//! This module provides the [`RenderPass`] and [`RenderPassBuilder`] types for
//! recording graphics commands using Vulkan's dynamic rendering feature
//! (`VK_KHR_dynamic_rendering`, core in Vulkan 1.3).
//!
//! # Overview
//!
//! Dynamic rendering simplifies the render pass model by eliminating the need
//! for `VkRenderPass` and `VkFramebuffer` objects. Instead, attachments are
//! specified directly when beginning rendering.
//!
//! [`vk::RenderPass`] has been deprecated in Vulkan 1.4. Dynamic rendering will
//! be the only way to organize draws in pumicite.
//!
//! # Usage
//!
//! ```ignore
//! encoder.begin_rendering()
//!     .render_area(IVec2::ZERO, UVec2::new(800, 600))
//!     .color_attachment(0, |a| {
//!         a.view(&color_view).clear([0.0, 0.0, 0.0, 1.0]).store(true);
//!     })
//!     .depth_attachment(|a| {
//!         a.view(&depth_view).clear(1.0).store(true);
//!     })
//!     .begin()
//!     .bind_pipeline(&pipeline)
//!     .set_viewport(0, &[viewport])
//!     .set_scissor(0, &[scissor])
//!     .draw(3, 1, 0, 0)
//!     .end();
//! ```
//!
//! # Continuing Render Passes
//!
//! Use [`continue_rendering`](CommandEncoder::continue_rendering) to resume an
//! already-active render pass on the command encoder.

use std::ops::{Deref, DerefMut, Range};

use ash::vk;
use glam::{IVec2, IVec4, UVec2, UVec4, Vec4};

use crate::{
    Device, HasDevice,
    buffer::BufferLike,
    command::RenderPassState,
    image::ImageViewLike,
    pipeline::{Pipeline, PipelineLayout},
    utils::{AsVkHandle, Version},
};

use super::CommandEncoder;

impl<'a> CommandEncoder<'a> {
    /// Begins a new dynamic render pass.
    ///
    /// Returns a [`RenderPassBuilder`] that allows configuring render area,
    /// color attachments, and depth/stencil attachments and other render pass attributes
    /// before entering the render pass.
    ///
    /// Note that on tile-based architectures, starting a new render pass may have performance implications.
    /// It is generally recommended to group multiple "subpasses" in the same render pass, separately by
    /// pipeline barriers with [`vk::DependencyFlags::BY_REGION`]. This ensures that framebuffer contents
    /// stay on-tile as much as possible, reducing bandwidth requirements.
    pub fn begin_rendering<'this>(&'this mut self) -> RenderPassBuilder<'this, 'a> {
        let info = vk::RenderingInfo {
            layer_count: 1,
            ..Default::default()
        };
        RenderPassBuilder {
            encoder: self,
            info,
            color_attachments: Default::default(),
            depth_attachment: None,
            stencil_attachment: None,
        }
    }

    /// Continue an already-active render pass.
    ///
    /// Returns `Some` if a render pass is active, `None` otherwise.
    pub fn continue_rendering<'this>(&'this mut self) -> Option<RenderPass<'this, 'a>> {
        if matches!(
            self.render_pass_state,
            RenderPassState::InsideRenderPass { .. }
        ) {
            Some(RenderPass { encoder: self })
        } else {
            None
        }
    }

    pub fn inside_renderpass(&self) -> bool {
        matches!(
            self.render_pass_state,
            RenderPassState::InsideRenderPass { .. }
        )
    }
}

/// Builder for configuring a dynamic render pass
///
/// Created by [`CommandEncoder::begin_rendering`]. Use the builder methods to
/// configure attachments and render area, then call [`begin`](Self::begin) to
/// start recording draw calls.
pub struct RenderPassBuilder<'encoder, 'a> {
    encoder: &'encoder mut CommandEncoder<'a>,
    info: vk::RenderingInfo<'a>,
    color_attachments: smallvec::SmallVec<[vk::RenderingAttachmentInfo<'a>; 3]>,
    depth_attachment: Option<vk::RenderingAttachmentInfo<'a>>,
    stencil_attachment: Option<vk::RenderingAttachmentInfo<'a>>,
}
impl<'encoder, 'a> RenderPassBuilder<'encoder, 'a> {
    /// Starts the render pass and returns a [`RenderPass`] for recording draw calls.
    ///
    /// After calling this, you can bind pipelines, set dynamic state, and issue
    /// draw commands. Call [`RenderPass::end`] when finished.
    pub fn begin(mut self) -> RenderPass<'encoder, 'a> {
        self.info.p_color_attachments = self.color_attachments.as_ptr();
        self.info.color_attachment_count = self.color_attachments.len() as u32;

        #[cfg(debug_assertions)]
        {
            for (i, info) in self.color_attachments.iter().enumerate() {
                use ash::vk::Handle;
                debug_assert!(
                    !info.image_view.is_null(),
                    "Color attachment {i} was unspecified",
                );
            }
        }
        if let Some(depth) = self.depth_attachment.as_ref() {
            self.info.p_depth_attachment = depth;
        }
        if let Some(stencil) = self.stencil_attachment.as_ref() {
            self.info.p_stencil_attachment = stencil;
        }
        unsafe {
            self.encoder
                .device()
                .cmd_begin_rendering(self.encoder.buffer().buffer, &self.info);
        }
        self.encoder.render_pass_state = super::RenderPassState::InsideRenderPass {
            render_area: self.info.render_area,
        };
        RenderPass {
            encoder: self.encoder,
        }
    }

    /// Sets the render area (the region of attachments that will be rendered to).
    pub fn render_area(mut self, offset: IVec2, extent: UVec2) -> Self {
        self.info.render_area.offset.x = offset.x;
        self.info.render_area.offset.y = offset.y;
        self.info.render_area.extent.width = extent.x;
        self.info.render_area.extent.height = extent.y;
        self
    }
    /// Set the number of layers rendered to in each attachment when viewMask is 0.
    pub fn layer_count(mut self, layer_count: u32) -> Self {
        self.info.layer_count = layer_count;
        self
    }
    /// Set a bitfield of view indices describing which views are active during rendering, when it is not 0.
    pub fn view_mask(mut self, layer_count: u32) -> Self {
        self.info.layer_count = layer_count;
        self
    }

    /// Configures the depth attachment for this render pass.
    ///
    /// The builder closure receives a [`RenderPassAttachmentBuilder`] to configure
    /// the attachment's image view, load/store operations, and clear value.
    pub fn depth_attachment(
        mut self,
        builder: impl FnOnce(RenderPassAttachmentBuilder<'_, 'a>),
    ) -> Self {
        if self.depth_attachment.is_none() {
            self.depth_attachment = Some(vk::RenderingAttachmentInfo {
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            });
        }
        builder(RenderPassAttachmentBuilder {
            device: self.encoder.device(),
            attachment: self.depth_attachment.as_mut().unwrap(),
        });
        self
    }

    /// Configures the stencil attachment for this render pass.
    ///
    /// The builder closure receives a [`RenderPassAttachmentBuilder`] to configure
    /// the attachment's image view, load/store operations, and clear value.
    pub fn stencil_attachment(
        mut self,
        builder: impl FnOnce(RenderPassAttachmentBuilder<'_, 'a>),
    ) -> Self {
        if self.stencil_attachment.is_none() {
            self.stencil_attachment = Some(vk::RenderingAttachmentInfo {
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            });
        }
        builder(RenderPassAttachmentBuilder {
            device: self.encoder.device(),
            attachment: self.depth_attachment.as_mut().unwrap(),
        });
        self
    }

    /// Configures a color attachment at the specified index.
    ///
    /// Multiple color attachments can be configured for multi-render-target (MRT)
    /// rendering. The `index` corresponds to `location` in fragment shader outputs.
    pub fn color_attachment(
        mut self,
        index: u32,
        builder: impl FnOnce(RenderPassAttachmentBuilder<'_, 'a>),
    ) -> Self {
        if index <= self.color_attachments.len() as u32 {
            self.color_attachments.extend(std::iter::repeat_n(
                vk::RenderingAttachmentInfo {
                    load_op: vk::AttachmentLoadOp::DONT_CARE,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                },
                index as usize - self.color_attachments.len() + 1,
            ));
        }
        builder(RenderPassAttachmentBuilder {
            device: self.encoder.device(),
            attachment: &mut self.color_attachments[index as usize],
        });
        self
    }
}

/// Builder for configuring a single render pass attachment.
///
/// Configure the attachment's image view, load/store operations, and optional
/// clear value or resolve target.
pub struct RenderPassAttachmentBuilder<'t, 'a> {
    device: &'t Device,
    attachment: &'t mut vk::RenderingAttachmentInfo<'a>,
}
impl HasDevice for RenderPassAttachmentBuilder<'_, '_> {
    fn device(&self) -> &Device {
        self.device
    }
}
impl<'a> RenderPassAttachmentBuilder<'_, 'a> {
    /// Sets the image view for this attachment.
    pub fn view(&mut self, image: &'a impl ImageViewLike) -> &mut Self {
        self.attachment.image_view = image.vk_handle();
        self
    }

    /// Sets the image layout used during rendering (defaults to `GENERAL`).
    pub fn image_layout(&mut self, layout: vk::ImageLayout) -> &mut Self {
        self.attachment.image_layout = layout;
        self
    }

    /// Sets the resolve mode for MSAA resolve operations.
    pub fn resolve_mode(&mut self, resolve_mode: vk::ResolveModeFlags) -> &mut Self {
        self.attachment.resolve_mode = resolve_mode;
        self
    }

    /// Sets the resolve target image view for MSAA attachments.
    pub fn resolve_view(&mut self, image: &'a impl ImageViewLike) -> &mut Self {
        self.attachment.resolve_image_view = image.vk_handle();
        self
    }

    /// Configures whether to load the attachment's previous contents.
    pub fn load(&mut self, should_load: bool) -> &mut Self {
        if should_load {
            self.attachment.load_op = vk::AttachmentLoadOp::LOAD;
        } else {
            self.attachment.load_op = vk::AttachmentLoadOp::NONE_KHR;
        }
        self
    }

    /// Configures whether to store the attachment's contents after rendering.
    ///
    /// If `true`, uses `STORE`. If `false`, uses `NONE` (contents discarded).
    pub fn store(&mut self, should_store: bool) -> &mut Self {
        if should_store {
            self.attachment.store_op = vk::AttachmentStoreOp::STORE;
        } else {
            self.attachment.store_op = vk::AttachmentStoreOp::NONE;
        }
        self
    }

    /// Clears the attachment to a specific value at the start of rendering.
    ///
    /// Accepts various clear value types implementing [`IntoClearValue`]:
    /// - `[f32; 4]` or `Vec4` for color
    /// - `f32` for depth
    /// - `vk::ClearDepthStencilValue` for combined depth/stencil
    pub fn clear(&mut self, clear_op: impl IntoClearValue) -> &mut Self {
        self.attachment.load_op = vk::AttachmentLoadOp::CLEAR;
        self.attachment.clear_value = clear_op.into_clear_value();
        self
    }
}

/// Trait for types that can be converted to a Vulkan clear value.
///
/// Implemented for common clear value types like `[f32; 4]`, `Vec4`, `f32`, etc.
pub trait IntoClearValue {
    fn into_clear_value(self) -> vk::ClearValue;
}
impl IntoClearValue for [f32; 4] {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue { float32: self },
        }
    }
}
impl IntoClearValue for Vec4 {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: self.into(),
            },
        }
    }
}
impl IntoClearValue for UVec4 {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue {
                uint32: self.into(),
            },
        }
    }
}
impl IntoClearValue for IVec4 {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue { int32: self.into() },
        }
    }
}

impl IntoClearValue for vk::ClearDepthStencilValue {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: self,
        }
    }
}
impl IntoClearValue for f32 {
    fn into_clear_value(self) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: self,
                stencil: 0,
            },
        }
    }
}

/// An active render pass for recording draw commands.
///
/// Created by [`RenderPassBuilder::begin`] or [`CommandEncoder::continue_rendering`].
/// Provides methods for binding pipelines, setting dynamic state, and issuing draw calls.
///
/// # Deref
///
/// `RenderPass` implements `Deref<Target = CommandEncoder>`, so you can call
/// encoder methods like `push_constants` directly on the render pass.
pub struct RenderPass<'a, 'b> {
    encoder: &'a mut CommandEncoder<'b>,
}
impl<'a, 'b> Deref for RenderPass<'a, 'b> {
    type Target = CommandEncoder<'b>;

    fn deref(&self) -> &Self::Target {
        self.encoder
    }
}
impl<'a, 'b> DerefMut for RenderPass<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoder
    }
}

impl<'a> RenderPass<'_, 'a> {
    /// Binds a graphics pipeline for subsequent draw commands.
    pub fn bind_pipeline(&mut self, pipeline: &'a Pipeline) {
        unsafe {
            self.encoder.device().cmd_bind_pipeline(
                self.encoder.buffer().buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.vk_handle(),
            );
        }
    }

    /// Binds vertex buffers starting at the specified binding index.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: impl IntoIterator<Item = &'a impl BufferLike>,
    ) {
        // most devices have up to 32 vertex bindings.
        let mut handles: smallvec::SmallVec<[vk::Buffer; 4]> = Default::default();
        let mut offsets: smallvec::SmallVec<[u64; 4]> = Default::default();
        let mut sizes: smallvec::SmallVec<[u64; 4]> = Default::default();
        for i in buffers.into_iter() {
            handles.push(i.vk_handle());
            offsets.push(i.offset());
            sizes.push(i.size());
        }
        unsafe {
            // If possible, call the v2 of cmd_bind_vertex_buffers.
            // The V1 version of this call may not take vertex buffer size into account,
            // causing false positives with the validation layer.
            // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8502#issuecomment-2423530399
            if self.encoder.device().instance().api_version() >= Version::V1_3 {
                self.encoder.device().cmd_bind_vertex_buffers2(
                    self.encoder.buffer().buffer,
                    first_binding,
                    &handles,
                    &offsets,
                    Some(&sizes),
                    None,
                );
            } else {
                self.encoder.device().cmd_bind_vertex_buffers(
                    self.encoder.buffer().buffer,
                    first_binding,
                    &handles,
                    &offsets,
                );
            }
        }
    }

    /// Binds an index buffer for indexed drawing.
    pub fn bind_index_buffer(
        &mut self,
        buffer: &'a impl BufferLike,
        offset: u64,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.encoder.device().cmd_bind_index_buffer(
                self.encoder.buffer().buffer,
                buffer.vk_handle(),
                buffer.offset() + offset,
                index_type,
            );
        }
    }

    /// Sets the viewport transformation for subsequent draw commands.
    pub fn set_viewport(&mut self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.encoder.device().cmd_set_viewport(
                self.encoder.buffer().buffer,
                first_viewport,
                viewports,
            );
        }
    }

    /// Sets the scissor rectangles for subsequent draw commands.
    pub fn set_scissor(&mut self, first_viewport: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.encoder.device().cmd_set_scissor(
                self.encoder.buffer().buffer,
                first_viewport,
                scissors,
            );
        }
    }

    /// Pushes a descriptor set directly for graphics commands.
    ///
    /// This is an alternative to `bind_descriptor_sets` that allows updating descriptors
    /// inline without allocating from a descriptor pool. Requires the
    /// `VK_KHR_push_descriptor` extension.
    pub fn push_descriptor_set(
        &mut self,
        layout: &PipelineLayout,
        set: u32,
        descriptor_writes: &[vk::WriteDescriptorSet<'_>],
    ) {
        debug_assert!(layout.device() == self.encoder.device());
        unsafe {
            self.encoder
                .device()
                .extension::<ash::khr::push_descriptor::Meta>()
                .cmd_push_descriptor_set(
                    self.encoder.buffer().buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout.vk_handle(),
                    set,
                    descriptor_writes,
                );
        }
    }

    /// Draws non-indexed primitives.
    pub fn draw(&mut self, vertex_range: Range<u32>, instance_range: Range<u32>) {
        unsafe {
            self.encoder.device().cmd_draw(
                self.encoder.buffer().buffer,
                vertex_range.end - vertex_range.start,
                instance_range.end - instance_range.start,
                vertex_range.start,
                instance_range.start,
            );
        }
    }

    /// Draws indexed primitives using the bound index buffer.
    pub fn draw_indexed(
        &mut self,
        index_range: Range<u32>,
        instance_range: Range<u32>,
        vertex_offset: i32,
    ) {
        unsafe {
            self.encoder.device().cmd_draw_indexed(
                self.encoder.buffer().buffer,
                index_range.end - index_range.start,
                instance_range.end - instance_range.start,
                index_range.start,
                vertex_offset,
                instance_range.start,
            );
        }
    }

    /// Draws multiple batches of non-indexed primitives.
    ///
    /// Uses `VK_EXT_multi_draw` if available, otherwise falls back to individual draw calls.
    pub fn draw_multi(&mut self, draws: &[vk::MultiDrawInfoEXT], instance_range: Range<u32>) {
        if let Ok(extension) = self
            .encoder
            .device()
            .get_extension::<ash::ext::multi_draw::Meta>()
        {
            unsafe {
                (extension.fp().cmd_draw_multi_ext)(
                    self.encoder.buffer().buffer,
                    draws.len() as u32,
                    draws.as_ptr(),
                    instance_range.end - instance_range.start,
                    instance_range.start,
                    std::mem::size_of::<vk::MultiDrawInfoEXT>() as u32,
                )
            }
        } else {
            for draw in draws {
                self.draw(
                    draw.first_vertex..(draw.first_vertex + draw.vertex_count),
                    instance_range.clone(),
                );
            }
        }
    }
    /// Draws multiple batches of non-indexed primitives.
    ///
    /// Uses `VK_EXT_multi_draw` if available, otherwise falls back to individual draw calls.
    ///
    /// `vertex_offsets`: If specified, [`vk::MultiDrawIndexedInfoEXT::vertex_offset`] will be ignored.
    /// all draw calls will be considered to have the same vertex_offset
    pub fn draw_multi_indexed(
        &mut self,
        draws: &[vk::MultiDrawIndexedInfoEXT],
        instance_range: Range<u32>,
        vertex_offsets: Option<i32>,
    ) {
        if let Ok(extension) = self
            .encoder
            .device()
            .get_extension::<ash::ext::multi_draw::Meta>()
        {
            unsafe {
                (extension.fp().cmd_draw_multi_indexed_ext)(
                    self.encoder.buffer().buffer,
                    draws.len() as u32,
                    draws.as_ptr(),
                    instance_range.end - instance_range.start,
                    instance_range.start,
                    12,
                    vertex_offsets
                        .as_ref()
                        .map(|x| x as *const i32)
                        .unwrap_or_default(),
                )
            }
        } else {
            for draw in draws.iter() {
                self.draw_indexed(
                    draw.first_index..(draw.first_index + draw.index_count),
                    instance_range.clone(),
                    draw.vertex_offset,
                );
            }
        }
    }

    /// Draws non-indexed primitives with draw parameters read from an indirect buffer on GPU timeline.
    ///
    /// `indirect_buffer` should contain `draw_count` number of [`vk::DrawIndirectCommand`] structs.
    pub fn draw_indirect(
        &mut self,
        indirect_buffer: &'a impl BufferLike,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.encoder.device().cmd_draw_indirect(
                self.encoder.buffer().buffer,
                indirect_buffer.vk_handle(),
                indirect_buffer.offset(),
                draw_count,
                stride,
            );
        }
    }

    /// Draws indexed primitives with draw parameters read from an indirect buffer on GPU timeline.
    ///
    /// `indirect_buffer` should contain `draw_count` number of [`vk::DrawIndexedIndirectCommand`] structs.
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a impl BufferLike,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.encoder.device().cmd_draw_indexed_indirect(
                self.encoder.buffer().buffer,
                indirect_buffer.vk_handle(),
                indirect_buffer.offset(),
                draw_count,
                stride,
            );
        }
    }

    /// Draws non-indexed primitives with both draw parameters and draw count read from an indirect buffer on GPU timeline.
    ///
    /// `indirect_buffer` should contain `draw_count` number of [`vk::DrawIndirectCommand`] structs.
    /// `count_buffer` contains a `u32` specifying the actual number of draws.
    pub fn draw_indirect_count(
        &mut self,
        indirect_buffer: &'a impl BufferLike,
        count_buffer: &'a impl BufferLike,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.encoder.device().cmd_draw_indirect_count(
                self.encoder.buffer().buffer,
                indirect_buffer.vk_handle(),
                indirect_buffer.offset(),
                count_buffer.vk_handle(),
                count_buffer.offset(),
                max_draw_count,
                stride,
            );
        }
    }

    /// Draws indexed primitives with draw count from a buffer.
    ///
    /// `indirect_buffer` should contain `draw_count` number of [`vk::DrawIndexedIndirectCommand`] structs.
    /// `count_buffer` contains a `u32` specifying the actual number of draws.
    pub fn draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &'a impl BufferLike,
        count_buffer: &'a impl BufferLike,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.encoder.device().cmd_draw_indexed_indirect_count(
                self.encoder.buffer().buffer,
                indirect_buffer.vk_handle(),
                indirect_buffer.offset(),
                count_buffer.vk_handle(),
                count_buffer.offset(),
                max_draw_count,
                stride,
            );
        }
    }

    /// Ends the current render pass.
    pub fn end(self) {
        unsafe {
            self.encoder
                .device()
                .cmd_end_rendering(self.encoder.buffer().buffer);
        }
        self.encoder.render_pass_state = super::RenderPassState::OutsideRenderPass;
    }

    pub fn render_area(&self) -> vk::Rect2D {
        match self.encoder.render_pass_state {
            RenderPassState::InsideRenderPass { render_area } => render_area,
            _ => panic!(),
        }
    }
}
