//! Buffer and image transfer commands.
//!
//! This module extends [`CommandEncoder`] with methods for copying data between
//! buffers and images, and for updating buffer contents.
use ash::vk;

use crate::{HasDevice, buffer::BufferLike, image::ImageLike};

use super::CommandEncoder;

impl<'a> CommandEncoder<'a> {
    /// Updates a buffer with inline data.
    ///
    /// This is a convenience method for small updates (≤64KB) that embeds the data
    /// directly in the command stream, avoiding the need for a staging buffer.
    pub fn update_buffer(&mut self, buffer: &'a impl BufferLike, data: &[u8]) {
        debug_assert!(!self.inside_renderpass());
        assert!(buffer.size() >= data.len() as u64);
        assert!(data.len() <= 65536);
        unsafe {
            self.device().cmd_update_buffer(
                self.buffer().buffer,
                buffer.vk_handle(),
                buffer.offset(),
                data,
            );
        }
    }

    /// Copies the entire contents of one buffer to another.
    ///
    /// The copy size is the minimum of the source and destination buffer sizes.
    pub fn copy_buffer(&mut self, src: &'a impl BufferLike, dst: &'a impl BufferLike) {
        self.copy_buffer_region(src, 0, dst, 0, src.size().min(dst.size()));
    }

    /// Copies a region from one buffer to another.
    pub fn copy_buffer_region(
        &mut self,
        src: &'a impl BufferLike,
        src_offset: u64,
        dst: &'a impl BufferLike,
        dst_offset: u64,
        size: u64,
    ) {
        debug_assert!(!self.inside_renderpass());
        unsafe {
            self.device().cmd_copy_buffer(
                self.buffer().buffer,
                src.vk_handle(),
                dst.vk_handle(),
                &[vk::BufferCopy {
                    src_offset: src.offset() + src_offset,
                    dst_offset: dst.offset() + dst_offset,
                    size,
                }],
            );
        }
    }
    pub fn copy_buffer_to_image_with(
        &mut self,
        buffer: &'a impl BufferLike,
        image: &'a impl ImageLike,
        copies: &[vk::BufferImageCopy],
    ) {
        self.copy_buffer_to_image_with_layout(buffer, image, copies, vk::ImageLayout::GENERAL);
    }

    /// Copies data from a buffer to an image.
    pub fn copy_buffer_to_image_with_layout(
        &mut self,
        buffer: &'a impl BufferLike,
        image: &'a impl ImageLike,
        copies: &[vk::BufferImageCopy],
        image_layout: vk::ImageLayout,
    ) {
        debug_assert!(!self.inside_renderpass());
        let copies_clone;
        let regions = if buffer.offset() > 0 {
            copies_clone = copies
                .iter()
                .cloned()
                .map(|x| vk::BufferImageCopy {
                    buffer_offset: x.buffer_offset + buffer.offset(),
                    ..x
                })
                .collect::<Vec<_>>();
            copies_clone.as_slice()
        } else {
            copies
        };
        unsafe {
            self.device().cmd_copy_buffer_to_image(
                self.buffer().buffer,
                buffer.vk_handle(),
                image.vk_handle(),
                image_layout,
                regions,
            );
        }
    }

    /// Blits (copies with scaling/conversion) image regions.
    ///
    /// Unlike a simple copy, blit can scale the image and convert between compatible
    /// formats. Both images must support blit operations for their formats.
    pub fn blit_image_with_layout(
        &mut self,
        src: &'a impl ImageLike,
        src_image_layout: vk::ImageLayout,
        dst: &'a impl ImageLike,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        debug_assert!(!self.inside_renderpass());
        unsafe {
            self.device().cmd_blit_image(
                self.buffer().buffer,
                src.vk_handle(),
                src_image_layout,
                dst.vk_handle(),
                dst_image_layout,
                regions,
                filter,
            );
        }
    }

    pub fn blit_image(
        &mut self,
        src: &'a impl ImageLike,
        dst: &'a impl ImageLike,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        self.blit_image_with_layout(
            src,
            vk::ImageLayout::GENERAL,
            dst,
            vk::ImageLayout::GENERAL,
            regions,
            filter,
        );
    }

    pub fn copy_image_to_image<S: ImageLike, T: ImageLike>(
        &mut self,
        src: &'a S,
        dst: &'a T,
        region: &[vk::ImageCopy],
    ) {
        self.copy_image_to_image_with_layout(
            src,
            vk::ImageLayout::GENERAL,
            dst,
            vk::ImageLayout::GENERAL,
            region,
        );
    }

    pub fn copy_image_to_image_with_layout<S: ImageLike, T: ImageLike>(
        &mut self,
        src: &'a S,
        src_layout: vk::ImageLayout,
        dst: &'a T,
        dst_layout: vk::ImageLayout,
        region: &[vk::ImageCopy],
    ) {
        unsafe {
            self.device().cmd_copy_image(
                self.buffer().buffer,
                src.vk_handle(),
                src_layout,
                dst.vk_handle(),
                dst_layout,
                region,
            );
        }
    }

    pub fn copy_image_to_buffer_with_layout(
        &mut self,
        src_image: &'a impl ImageLike,
        src_image_layout: vk::ImageLayout,
        dst_buf: &'a impl BufferLike,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device().cmd_copy_image_to_buffer(
                self.buffer().buffer,
                src_image.vk_handle(),
                src_image_layout,
                dst_buf.vk_handle(),
                regions,
            );
        }
    }

    pub fn clear_color_image<T: ImageLike>(
        &mut self,
        image: &'a T,
        clear_color: &vk::ClearColorValue,
    ) {
        self.clear_color_image_with_layout(image, clear_color, vk::ImageLayout::GENERAL);
    }

    /// Clears a color image to a solid color.
    pub fn clear_color_image_with_layout<T: ImageLike>(
        &mut self,
        image: &'a T,
        clear_color: &vk::ClearColorValue,
        image_layout: vk::ImageLayout,
    ) {
        debug_assert!(!self.inside_renderpass());
        unsafe {
            self.device().cmd_clear_color_image(
                self.buffer().buffer,
                image.vk_handle(),
                image_layout,
                clear_color,
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: image.mip_level_count(),
                    base_array_layer: 0,
                    layer_count: image.array_layer_count(),
                }],
            );
        }
    }
}
