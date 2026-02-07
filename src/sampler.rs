//! Vulkan sampler management.
//!
//! This module provides the [`Sampler`] type for texture sampling in shaders.
//!
//! # Overview
//!
//! Samplers define how textures are sampled, including filtering modes,
//! address wrapping, and mipmap selection. They are immutable after creation.
//!
//! # Bindless Integration
//!
//! When bindless is enabled on the device, samplers are automatically registered
//! in the bindless heap and can be accessed by index in shaders.

use crate::{Device, HasDevice, utils::AsVkHandle};
use ash::{VkResult, vk};
use std::fmt::Debug;

/// A Vulkan sampler for texture filtering.
///
/// Samplers control how image data is read in shaders, including:
/// - Filtering (nearest, linear, anisotropic)
/// - Address mode (repeat, clamp, mirror)
/// - Mipmap selection and LOD clamping
/// - Border color for clamp-to-border mode
///
/// # Bindless
///
/// If bindless is enabled, the sampler is automatically added to the bindless heap.
/// Use [`resource_handle`](Self::resource_handle) to get the index for shader access.
pub struct Sampler {
    device: Device,
    handle: vk::Sampler,
}
impl Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}
impl HasDevice for Sampler {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl Sampler {
    /// Creates a new sampler.
    pub fn new(device: Device, info: &vk::SamplerCreateInfo) -> VkResult<Self> {
        let inner = unsafe { device.create_sampler(info, None) }?;
        Ok(Self {
            device,
            handle: inner,
        })
    }
}

impl AsVkHandle for Sampler {
    type Handle = vk::Sampler;

    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.destroy_sampler(self.handle, None) }
    }
}
