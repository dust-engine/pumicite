//! Vulkan debug utilities integration.
//!
//! This module provides [`DebugUtilsPlugin`] for enabling Vulkan debug messaging,
//! which routes validation layer messages to Rust's tracing framework.

use bevy_app::Plugin;
use pumicite::ash::ext::debug_utils::Meta as DebugUtilsExt;

use pumicite::{Instance, debug::DebugUtilsMessenger};

use super::PumiciteApp;

/// Plugin that enables Vulkan debug messaging.
///
/// This is an **instance plugin** and must be added **before**
/// [`PumicitePlugin`](crate::PumicitePlugin).
///
/// # What It Does
///
/// 1. Enables the `VK_EXT_debug_utils` instance extension
/// 2. Adding a [`DebugUtilsMessenger`] resource that routes Vulkan messages to
///    the [`tracing`] logging framework.
///
/// # Production Use
///
/// Debug utils have minimal overhead when validation layers aren't loaded. Feel
/// free to leave this plugin enabled in production builds, but the choice is yours.
///
/// # Included in [`DefaultPlugins`](crate::DefaultPlugins)
///
/// This plugin is included by default when using `DefaultPlugins`.
#[derive(Default)]
pub struct DebugUtilsPlugin;

impl Plugin for DebugUtilsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_instance_extension::<DebugUtilsExt>().ok();
    }
    fn finish(&self, app: &mut bevy_app::App) {
        let instance: Instance = app.world().resource::<Instance>().clone();
        app.insert_resource(DebugUtilsMessenger::new(instance).unwrap());
    }
}
