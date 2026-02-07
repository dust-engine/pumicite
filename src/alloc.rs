//! GPU memory allocation.
//!
//! This module provides the [`Allocator`] type, a wrapper around the Vulkan Memory
//! Allocator (VMA) library for efficient GPU memory management.
//!
//! # Overview
//!
//! VMA handles the complexity of Vulkan memory allocation by:
//! - Pooling allocations to reduce API overhead
//! - Selecting appropriate memory types automatically
//! - Managing memory defragmentation
//!
//! # Usage
//!
//! Create an allocator once per device and pass it to buffer/image creation functions:
//!
//! ```
//! # use pumicite::{Device, Allocator, buffer::Buffer, ash::vk};
//! # let (device, queue) = Device::create_system_default().unwrap();
//! let allocator = Allocator::new(device.clone()).unwrap();
//! let buffer = Buffer::new_private(allocator, 1024, 4, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
//! ```

use std::{ops::Deref, sync::Arc};

use ash::{VkResult, vk};

use crate::{Device, HasDevice, utils::AsVkHandle};

/// A GPU memory allocator using the Vulkan Memory Allocator (VMA) library.
///
/// This is a reference-counted wrapper around VMA that handles efficient GPU
/// memory allocation. It automatically detects and enables buffer device address
/// support when available.
///
/// The allocator is thread-safe and can be cloned cheaply.
#[derive(Clone)]
pub struct Allocator(Arc<AllocatorInner>);
struct AllocatorInner {
    device: Device,
    inner: vk_mem::Allocator,
}

impl HasDevice for Allocator {
    fn device(&self) -> &Device {
        &self.0.device
    }
}

impl Allocator {
    /// Creates a new allocator for the given device.
    ///
    /// Automatically enables buffer device address support if the device has
    /// the `bufferDeviceAddress` feature enabled.
    pub fn new(device: Device) -> VkResult<Self> {
        let mut info = vk_mem::AllocatorCreateInfo::new(
            device.instance(),
            &device,
            device.physical_device().vk_handle(),
        );

        let buffer_device_address_enabled = device
            .feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>()
            .map(|f| f.buffer_device_address)
            .map(|b| b == vk::TRUE)
            .unwrap_or(false);
        if buffer_device_address_enabled {
            info.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }
        let alloc = unsafe { vk_mem::Allocator::new(info)? };
        Ok(Self(Arc::new(AllocatorInner {
            device,
            inner: alloc,
        })))
    }
}

impl Deref for Allocator {
    type Target = vk_mem::Allocator;

    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}
