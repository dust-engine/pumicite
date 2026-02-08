//! Bevy ECS integration for the Pumicite Vulkan abstraction library.
//!
//! `bevy_pumicite` provides a complete Vulkan rendering backend for Bevy applications,
//! replacing Bevy's default `wgpu`-based renderer with direct Vulkan access through
//! the [`pumicite`] abstraction layer.
//!
//! # Key Features
//!
//! - **Vulkan Instance & Device Management**: Automatic setup of Vulkan instance, physical
//!   device selection, and logical device creation with configurable extensions and features.
//!
//! - **Queue Management**: Flexible queue family selection with automatic aliasing when
//!   dedicated queues aren't available. Includes predefined queue types for rendering,
//!   compute, and transfer operations.
//!
//! - **Swapchain Integration**: Automatic swapchain creation and management tied to Bevy
//!   windows, with support for HDR output and configurable present modes.
//!
//! - **Asset Loading**: Built-in asset loaders for SPIR-V shaders, pipeline configurations
//!   (RON format), and textures (DDS, PNG, JPEG).
//!
//! - **Ray Tracing (RTX)**: Full support for Vulkan ray tracing pipelines, including BLAS/TLAS
//!   acceleration structure management and shader binding table generation.
//!
//! - **Staging & Transfers**: Ring buffer allocators and async transfer infrastructure for
//!   efficient GPU uploads.
//!
//! # Quick Start
//!
//! Add bevy_pumicite and bevy into your `Cargo.toml`. Disable `default-features` on `bevy`
//! (so that bevy_render doesn't get pulled into your dependency tree), but keep `bevy_winit`, `bevy_asset`
//! and `multi_threaded` enabled at a minimum.
//! ```toml
//! bevy = { version = "0.17.0-dev", default-features = false, features=["bevy_winit", "bevy_asset", "multi_threaded"] }
//! bevy_pumicite = "0.1"
//! ```
//!
//! The easiest way to get started is using [`DefaultPlugins`], which replaces Bevy's
//! rendering plugins with Pumicite equivalents:
//!
//! ```no_run
//! use bevy::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(bevy_pumicite::DefaultPlugins)
//!         .run();
//! }
//! ```
//!
//! # Plugin Ordering
//!
//! [`PumicitePlugin`] configures the Vulkan [Device](pumicite::ash::Device).
//! When manually configuring plugins, ordering is critical:
//!
//! 1. **Instance plugins** (e.g., [`SurfacePlugin`], [`DebugUtilsPlugin`]) must be added
//!    **before** [`PumicitePlugin`] because they configure the Vulkan instance, which must be
//!    prior to device creation.
//!
//! 2. **Device plugins** (e.g., RTX plugins, custom extension plugins) must be added
//!    **after** [`PumicitePlugin`] because they configure the Vulkan device.
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_pumicite::{PumicitePlugin, SurfacePlugin, DebugUtilsPlugin};
//! fn main() {
//!     App::new()
//!         .add_plugins(bevy::DefaultPlugins)
//!         // Instance plugins BEFORE PumicitePlugin
//!         .add_plugins(SurfacePlugin::default())
//!         .add_plugins(DebugUtilsPlugin::default())
//!         // Main plugin creates instance and device
//!         .add_plugins(PumicitePlugin::default())
//!         // Device plugins AFTER PumicitePlugin
//!         .add_plugins(bevy_pumicite::swapchain::SwapchainPlugin)
//!         .run();
//! }
//!
//! ```
#![feature(maybe_uninit_array_assume_init)]

mod debug;
pub mod loader;
mod pass;
mod plugin;
mod queue;
pub mod rtx;
pub mod shader;
pub mod staging;
mod surface;
pub mod swapchain;
mod system;
pub use pumicite;
mod heap;

pub use debug::DebugUtilsPlugin;
pub use heap::DescriptorHeap;
pub use plugin::{
    DefaultComputeSet, DefaultRenderSet, DefaultTransferSet, PumiciteApp, PumicitePlugin,
};
pub use queue::Queue;
pub use surface::SurfacePlugin;
pub use system::RenderState;

/// Convenience re-exports for the most commonly used types.
pub mod prelude {
    pub use pumicite::prelude::*;

    pub use crate::{
        DescriptorHeap,
        PumiciteApp,
        DefaultComputeSet,
        DefaultRenderSet,
        DefaultTransferSet,
        RenderState,
        shader::{ComputePipeline, GraphicsPipeline},
        staging::{
            BufferInitializer, DeviceLocalRingBuffer, HostVisibleRingBuffer, UniformRingBuffer,
        },
        swapchain::{SwapchainConfig, SwapchainImage},
    };
}

/// A plugin group that configures Bevy with Pumicite as the rendering backend.
///
/// This is equivalent to [`bevy::DefaultPlugins`] but with
/// Bevy's rendering plugins replaced by Pumicite equivalents.
///
/// # Disabled Bevy Plugins
///
/// The following Bevy plugins are disabled since Pumicite provides its own implementations:
/// - `bevy::render::RenderPlugin`
/// - `bevy_pbr::PbrPlugin`
/// - `bevy_core_pipeline::CorePipelinePlugin`
///
/// # Included Pumicite Plugins
///
/// - [`SurfacePlugin`] - Creates Vulkan surfaces for windows (instance plugin)
/// - [`DebugUtilsPlugin`] - Enables Vulkan debug messaging
/// - [`PumicitePlugin`] - Core plugin that creates the Vulkan instance and device
/// - [`SwapchainPlugin`](swapchain::SwapchainPlugin) - Manages swapchain creation and presentation
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::DefaultPlugins;
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .run();
/// ```
pub struct DefaultPlugins;

impl bevy_app::PluginGroup for DefaultPlugins {
    fn build(self) -> bevy_app::PluginGroupBuilder {
        bevy_internal::DefaultPlugins
            .build()
            .disable_plugin_named("bevy_render::RenderPlugin")
            .disable_plugin_named("bevy_pbr::PbrPlugin")
            .disable_plugin_named("bevy_core_pipeline::CorePipelinePlugin")
            .add(crate::SurfacePlugin)
            .add(crate::DebugUtilsPlugin)
            .add(crate::PumicitePlugin::default())
            .add(crate::swapchain::SwapchainPlugin)
    }
}
