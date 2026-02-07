//! # Pumicite
//!
//! A modern, high-performance Vulkan graphics library for Rust.
//!
//! Pumicite provides safe, ergonomic abstractions over Vulkan while maintaining
//! direct access to low-level GPU features. It integrates seamlessly with the
//! Bevy game engine but can also be used standalone.
//!
//! ## Quick Start
//!
//! ```
//! use pumicite::prelude::*;
//!
//! // Create a device with sensible defaults
//! let (device, mut queue) = Device::create_system_default().unwrap();
//! let allocator = Allocator::new(device.clone()).unwrap();
//!
//! // Create a buffer
//! let buffer = Buffer::new_upload(allocator, 1024, 4, vk::BufferUsageFlags::VERTEX_BUFFER).unwrap();
//! ```
//!
//! ## Overview
//!
//! ### Device and Instance
//!
//! The [`Instance`] is your connection to the Vulkan loader. Use [`Device`] to
//! create GPU resources and submit work.
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, Device};
//! let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! let mut builder = Instance::builder(entry);
//! builder.enable_extension::<ash::ext::debug_utils::Meta>().ok();
//! let instance = builder.build().unwrap();
//!
//! let pdevice = instance.enumerate_physical_devices().unwrap().next().unwrap();
//! let mut device_builder = Device::builder(pdevice);
//! device_builder.enable_queue(/* queue family */ 0, /* priority */ 1.0);
//! let device = device_builder.build().unwrap();
//! ```
//!
//! ### Memory Allocation
//!
//! Pumicite uses [`Allocator`] VMA for efficient GPU memory management. To create GPU resources,
//! pass in the allocator as the first argument:
//!
//! - [`Buffer::new_private`](buffer::Buffer::new_private) - GPU-only device local buffers
//! - [`Buffer::new_upload`](buffer::Buffer::new_upload) - CPU-writonly device local buffers
//! - [`Buffer::new_dynamic`](buffer::Buffer::new_dynamic) - Frequently updated data
//! - [`Image::new_private`](image::Image::new_private) - GPU-only device local images
//! - [`Image::new_upload`](image::Image::new_upload) - CPU-writonly device local images
//!
//! ### Command Recording
//!
//! Commands are recorded into [`CommandBuffer`](command::CommandBuffer)s and
//! submitted via [`Queue`]:
//!
//! ```
//! # use pumicite::{Device, command::CommandPool, sync::Timeline};
//! # let (device, mut queue) = Device::create_system_default().unwrap();
//! let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
//! let mut timeline = Timeline::new(device.clone()).unwrap();
//! let mut cmd = pool.alloc().unwrap();
//!
//! pool.begin(&mut cmd).unwrap();
//! timeline.schedule(&mut cmd);
//!
//! pool.record(&mut cmd, |encoder| {
//!     // Record commands using encoder
//! });
//!
//! pool.finish(&mut cmd).unwrap();
//! queue.submit(&mut cmd).unwrap();
//! ```
//!
//! ### Synchronization
//!
//! Pumicite uses timeline semaphores for synchronization between submissions:
//!
//! - [`Queue`] - Submits work (start order guaranteed, not completion order)
//! - [`Timeline`](sync::Timeline) - Enforces sequential execution across submissions
//! - [`GPUMutex`](sync::GPUMutex) - Protects resources from concurrent CPU/GPU access
//!
//! ### Bindless Resources
//!
//! Enable bindless rendering for descriptor indexing:
//!
//! ```
//! # use std::sync::Arc;
//! # use pumicite::{Instance, Device, bindless::BindlessConfig};
//! # let entry = Arc::new(unsafe { ash::Entry::load() }.unwrap());
//! # let instance = Instance::builder(entry).build().unwrap();
//! # let pdevice = instance.enumerate_physical_devices().unwrap().next().unwrap();
//! let mut builder = Device::builder(pdevice);
//! builder.enable_bindless(BindlessConfig::default()).unwrap();
//! # builder.enable_queue(0, 1.0);
//! let device = builder.build().unwrap();
//!
//! let heap = device.bindless_heap();
//! // Use heap.add_resource() to add images/buffers for shader access
//! ```
//!
//! ## Feature Flags
//!
//! - `bevy` - Enables Bevy ECS integration with the `bevy` module
//!
//! ## Requirements
//!
//! - Vulkan 1.2+ with Timeline Semaphore and Synchronization2 extensions
//! - Rust nightly

#![feature(min_specialization)]
#![feature(ptr_metadata)]
#![feature(box_as_ptr)]
#![feature(atomic_try_update)]
#![feature(async_fn_traits, unboxed_closures)]

mod alloc;
pub mod buffer;
pub mod command;
pub mod debug;
pub mod descriptor;
pub mod device;
mod extension;
mod future;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod pipeline;
mod queue;
pub mod rtx;
mod sampler;
mod surface;
pub mod swapchain;
pub mod sync;
pub mod tracking;
pub mod utils;

pub use alloc::Allocator;
pub use device::{Device, HasDevice};
pub use extension::*;
pub use instance::Instance;
pub use queue::Queue;
pub use sampler::Sampler;
pub use surface::Surface;
pub mod bindless;

#[cfg(feature = "bevy")]
pub mod bevy;

pub use ash;

pub mod prelude {
    pub use crate::{
        Allocator, Device, HasDevice, ash,
        ash::vk,
        buffer::{Buffer, BufferExt, BufferLike},
        command::{CommandBuffer, CommandEncoder},
        debug::DebugObject,
        image::{Image, ImageExt, ImageLike},
        queue::Queue,
        sync::GPUMutex,
        tracking::{Access, ResourceState},
        utils::AsVkHandle,
    };
}
