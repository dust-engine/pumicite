//! Vulkan debug utilities integration.
//!
//! This module provides [`DebugUtilsPlugin`] for enabling Vulkan debug messaging,
//! which routes validation layer messages to Rust's tracing framework.
//! [`DeviceFaultPlugin`] enables driver fault details for device losses.

use bevy_app::{Plugin, Startup};
use bevy_ecs::prelude::*;
use pumicite::ash::ext::debug_utils::Meta as DebugUtilsExt;
use pumicite::ash::ext::device_fault::Meta as DeviceFaultExt;
use pumicite::ash::vk;

use pumicite::{Instance, debug::DebugUtilsMessenger, device::DeviceBuilder};

use super::{CreateDevice, PumiciteApp};

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

/// Plugin that enables `VK_EXT_device_fault` and its `deviceFault` feature when the
/// device supports them.
///
/// This is a **device plugin** and may be added in any order relative to
/// [`PumicitePlugin`](crate::PumicitePlugin).
///
/// With it enabled, a [`DeviceError`] for `ERROR_DEVICE_LOST` includes the driver's
/// [`DeviceFaultReport`] (faulting GPU addresses and vendor fault codes) when it is
/// formatted, so unwrapping the [`DeviceResult`] of a lost submission, present,
/// or GPU wait prints the report in the panic message.
///
/// [`DeviceError`]: pumicite::fault::DeviceError
/// [`DeviceResult`]: pumicite::fault::DeviceResult
/// [`DeviceFaultReport`]: pumicite::fault::DeviceFaultReport
#[derive(Default)]
pub struct DeviceFaultPlugin;

impl Plugin for DeviceFaultPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Startup, enable_device_fault.before(CreateDevice));
    }
}

fn enable_device_fault(mut device_builder: ResMut<DeviceBuilder>) {
    if device_builder.enable_extension::<DeviceFaultExt>().is_ok() {
        device_builder
            .enable_feature::<vk::PhysicalDeviceFaultFeaturesEXT>(|f| &mut f.device_fault)
            .ok();
    }
}
