/*!
Vulkan swapchain management for presenting rendered images to windows.

A swapchain is a queue of images that can be presented to a surface. This module
handles swapchain creation, image acquisition, and presentation for Bevy windows.

# Swapchain Image Lifecycle

1. **Creation**: When a window with a surface is created, a swapchain is automatically
   created with default or custom [`SwapchainConfig`].

2. **Acquisition**: Before rendering each frame, an image is acquired from the swapchain
   via [`acquire_swapchain_image`].

3. **Rendering**: Systems in [`SwapchainSet`] can access the current image via
   [`SwapchainImage`] and render to it.

4. **Presentation**: After rendering, the image is presented to the display via
   [`present`].

5. **Recreation**: The swapchain is automatically recreated when the window is resized
   or becomes suboptimal.

# Configuration

Add a [`SwapchainConfig`] component to a window entity to customize swapchain behavior:

```no_run
use bevy::prelude::*;
use bevy_pumicite::pumicite::ash::vk;
fn main() {
   let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.run();
}
```

# System Ordering
Every system that needs to access the swapchain image stored in the [`SwapchainImage`] component needs to be ordered
to run in set [`SwapchainSet`]. This ensures your system runs after image acquisition and before presentation.
```no_run
# use bevy::prelude::*;
# fn my_render_system(){}
use bevy_pumicite::swapchain::SwapchainSet;
App::new().add_systems(PostUpdate, my_render_system.in_set(SwapchainSet));
```
*/

use std::collections::BTreeSet;

use bevy_app::{App, Plugin};
use bevy_ecs::{prelude::*, query::QueryFilter};
use bevy_window::{PrimaryWindow, Window};
use glam::UVec2;
use pumicite::ash::ext::surface_maintenance1::Meta as ExtSurfaceMaintenance1;
use pumicite::ash::vk;
use pumicite::{
    ash::ext::swapchain_maintenance1::Meta as ExtSwapchainMaintenance1,
    ash::khr::swapchain_mutable_format::Meta as ExtSwapchainMutableFormat, tracking::ResourceState,
};
use pumicite::{ash::khr::swapchain::Meta as KhrSwapchain, swapchain::HDRMode};

use crate::queue::RenderQueue;

use super::{PumiciteApp, queue::Queue};
use pumicite::{
    Device, HasDevice, Surface,
    physical_device::PhysicalDevice,
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::GPUMutex,
    utils::SharingMode,
};

/// System set for rendering to the swapchain image.
///
/// Systems in this set run between swapchain image acquisition and presentation,
/// giving them access to the current [`SwapchainImage`].
///
/// # Ordering
///
/// - Runs **after** `acquire_swapchain_image` (image is ready after a call to `vkAcquireNextImageKHR`)
/// - Runs **before** `present` (image will be presented with a call to `vkQueuePresentKHR`)
///
/// Every system that needs to access the swapchain image stored in the [`SwapchainImage`] component needs to be ordered
/// to run in set [`SwapchainSet`]. This ensures your system runs after image acquisition and before presentation.
/// ```ignore
///# use bevy::prelude::*;
///# fn my_render_system(){}
///  use bevy_pumicite::DefaultRenderSet;
///  App::new().add_systems(PostUpdate, my_render_system.in_set(DefaultRenderSet));
/// ```
#[derive(SystemSet, Hash, PartialEq, Eq, Debug, Clone)]
pub struct SwapchainSet;

/// Plugin that manages swapchain creation and presentation.
///
/// This is a **device plugin** and should be added **after** [`PumicitePlugin`](crate::PumicitePlugin).
///
/// # Included in [`DefaultPlugins`](crate::DefaultPlugins)
pub struct SwapchainPlugin;
impl Plugin for SwapchainPlugin {
    fn build(&self, app: &mut App) {
        app.add_device_extension::<KhrSwapchain>().unwrap();

        app.add_device_extension::<ExtSurfaceMaintenance1>().ok();
        app.add_device_extension::<ExtSwapchainMaintenance1>().ok();
        app.add_device_extension::<ExtSwapchainMutableFormat>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>(|x| {
            &mut x.swapchain_maintenance1
        })
        .ok();

        app.add_systems(
            bevy_app::PostUpdate,
            (
                extract_swapchains.after(super::surface::extract_surfaces),
                // Always present on the Graphics queue.
                // TODO: this isn't exactly the best. Ideally we check surface-pdevice-queuefamily compatibility
                // using vkGetPhysicalDeviceSurfaceSupportKHR select the best one.
                acquire_swapchain_image::<With<PrimaryWindow>>,
                present::<With<PrimaryWindow>>,
            )
                .chain(),
        );
        app.configure_sets(
            bevy_app::PostUpdate,
            SwapchainSet
                .before(present::<With<PrimaryWindow>>)
                .after(acquire_swapchain_image::<With<PrimaryWindow>>),
        );

        app.add_message::<SuboptimalMessage>();
    }
}

/// Configuration for swapchain creation.
///
/// Add this component to a window entity to customize swapchain behavior.
/// If not present, defaults are used.
///
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::pumicite::ash::vk;
/// fn main() {
///    let mut app = bevy::app::App::new();
///     app.add_plugins(bevy_pumicite::DefaultPlugins);
///     let primary_window = app
///         .world_mut()
///         .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
///         .iter(app.world())
///         .next()
///         .unwrap();
///     app.world_mut()
///         .entity_mut(primary_window)
///         .insert(bevy_pumicite::swapchain::SwapchainConfig {
///             image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
///             ..Default::default()
///         });
///     app.run();
/// }
/// ```
#[derive(Component)]
pub struct SwapchainConfig {
    /// Additional swapchain creation flags.
    pub flags: vk::SwapchainCreateFlagsKHR,

    /// Minimum number of swapchain images. Higher values reduce tearing but
    /// increase latency. Common values: 2 (double buffer), 3 (triple buffer).
    ///
    /// Default: 3
    pub min_image_count: u32,

    /// Explicit surface format (format + color space). If `None`, the implementation
    /// selects the best available format based on [`hdr`](Self::hdr) setting.
    pub image_format: Option<vk::SurfaceFormatKHR>,

    /// Number of image array layers. Use 1 for non-stereoscopic rendering.
    ///
    /// Default: 1
    pub image_array_layers: u32,

    /// How the swapchain images will be used. Must include at least `COLOR_ATTACHMENT`.
    ///
    /// Default: `COLOR_ATTACHMENT`
    pub image_usage: vk::ImageUsageFlags,

    /// Required format feature flags for the selected format.
    pub required_feature_flags: vk::FormatFeatureFlags,

    /// Queue family sharing mode for multi-queue access.
    ///
    /// Default: `Exclusive`
    pub sharing_mode: SharingMode<Vec<u32>>,

    /// Surface transform to apply (rotation, mirroring).
    ///
    /// Default: `IDENTITY`
    pub pre_transform: vk::SurfaceTransformFlagsKHR,

    /// Whether pixels outside the visible region can be discarded.
    /// Setting to `true` may improve performance.
    ///
    /// Default: `true`
    pub clipped: bool,

    /// HDR mode preference. Only used when [`image_format`](Self::image_format) is `None`.
    ///
    /// - `Off`: Prefer SDR formats (sRGB)
    /// - `Auto`: Select HDR if available
    /// - `Required`: Require HDR, fail if unavailable
    ///
    /// Default: `Off`
    pub hdr: HDRMode,

    /// Queue family index for pre-present layout transitions. Should match
    /// the queue family used for presentation (typically 0 for graphics).
    ///
    /// Default: 0
    pub queue_family_index: u32,
}
impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: None,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            required_feature_flags: vk::FormatFeatureFlags::empty(),
            sharing_mode: SharingMode::Exclusive,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            clipped: true,
            hdr: HDRMode::Off,
            queue_family_index: 0,
        }
    }
}

#[derive(Message)]
pub struct SuboptimalMessage {
    window: Entity,
}

fn recreate_swapchain(
    swapchain: &mut Swapchain,
    surface: &Surface,
    window: &Window,
    config: Option<&SwapchainConfig>,
) {
    let default_config = SwapchainConfig::default();
    let config = config.unwrap_or(&default_config);
    let create_info = get_create_info(
        surface,
        swapchain.device().physical_device(),
        window,
        config,
    );
    swapchain.recreate(&create_info).unwrap();
}

pub(super) fn extract_swapchains(
    mut commands: Commands,
    device: Res<Device>,
    mut window_created_messages: MessageReader<bevy_window::WindowCreated>,
    mut window_resized_messages: MessageReader<bevy_window::WindowResized>,
    mut suboptimal_messages: MessageReader<SuboptimalMessage>,
    mut query: Query<(
        &Window,
        Option<&SwapchainConfig>,
        Option<&mut Swapchain>,
        &Surface,
    )>,
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: Option<
        NonSend<bevy_ecs::system::NonSendMarker>,
    >,
) {
    // https://gist.github.com/nanokatze/bb03a486571e13a7b6a8709368bd87cf#file-handling-window-resize-md
    let mut windows_to_rebuild: BTreeSet<Entity> = BTreeSet::new();
    windows_to_rebuild.extend(window_resized_messages.read().filter_map(|message| {
        let (window, _, swapchain, _) = query.get(message.window).ok()?;
        let swapchain = swapchain?;
        if window.physical_height() != swapchain.extent().y
            || window.physical_width() != swapchain.extent().x
        {
            Some(message.window)
        } else {
            None
        }
    }));
    windows_to_rebuild.extend(suboptimal_messages.read().map(|a| a.window));
    for resized_window in windows_to_rebuild.into_iter() {
        let (window, config, swapchain, surface) = query.get_mut(resized_window).unwrap();
        if let Some(mut swapchain) = swapchain {
            recreate_swapchain(&mut swapchain, surface, window, config);
        }
    }
    for create_message in window_created_messages.read() {
        let (window, config, swapchain, surface) = query.get(create_message.window).unwrap();
        assert!(swapchain.is_none());
        let default_config = SwapchainConfig::default();
        let swapchain_config = config.unwrap_or(&default_config);
        let create_info =
            get_create_info(surface, device.physical_device(), window, swapchain_config);
        tracing::info!(
            "Swapchain created with {:?} and {:?}",
            create_info.image_format,
            create_info.image_color_space
        );
        let new_swapchain =
            Swapchain::create(device.clone(), surface.clone(), &create_info).unwrap();
        commands
            .entity(create_message.window)
            .insert(new_swapchain)
            .insert(SwapchainImage {
                inner: None,
                state: ResourceState::default(),
            });
    }
}

fn get_create_info<'a>(
    surface: &'_ Surface,
    pdevice: &'_ PhysicalDevice,
    window: &'_ Window,
    config: &'a SwapchainConfig,
) -> SwapchainCreateInfo<'a> {
    let surface_capabilities = pdevice.get_surface_capabilities(surface).unwrap();
    let supported_present_modes = pdevice.get_surface_present_modes(surface).unwrap();
    let image_format = config.image_format.unwrap_or_else(|| {
        pumicite::swapchain::get_surface_preferred_format(
            surface,
            pdevice,
            config.required_feature_flags,
            config.hdr,
        )
        .expect("Unable to find suitable surface format")
    });
    let mut image_extent = UVec2::new(
        surface_capabilities.current_extent.width,
        surface_capabilities.current_extent.height,
    );
    if image_extent == UVec2::splat(u32::MAX) {
        // currentExtent is the current width and height of the surface, or the special value (0xFFFFFFFF, 0xFFFFFFFF)
        // indicating that the surface size will be determined by the extent of a swapchain targeting the surface.
        image_extent = UVec2::new(
            window.resolution.physical_width(),
            window.resolution.physical_height(),
        );
    }
    image_extent = image_extent.min(UVec2::new(
        surface_capabilities.max_image_extent.width,
        surface_capabilities.max_image_extent.height,
    ));
    image_extent = image_extent.max(UVec2::new(
        surface_capabilities.min_image_extent.width,
        surface_capabilities.min_image_extent.height,
    ));
    SwapchainCreateInfo {
        flags: config.flags,
        min_image_count: config
            .min_image_count
            .max(surface_capabilities.min_image_count)
            .min({
                if surface_capabilities.max_image_count == 0 {
                    u32::MAX
                } else {
                    surface_capabilities.max_image_count
                }
            }),
        image_format: image_format.format,
        image_color_space: image_format.color_space,
        image_extent,
        image_array_layers: config.image_array_layers,
        image_usage: config.image_usage,
        image_sharing_mode: match &config.sharing_mode {
            SharingMode::Exclusive => SharingMode::Exclusive,
            SharingMode::Concurrent {
                queue_family_indices,
            } => SharingMode::Concurrent {
                queue_family_indices,
            },
        },
        pre_transform: config.pre_transform,
        composite_alpha: match window.composite_alpha_mode {
            bevy_window::CompositeAlphaMode::Auto => {
                if surface_capabilities
                    .supported_composite_alpha
                    .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
                {
                    vk::CompositeAlphaFlagsKHR::OPAQUE
                } else {
                    vk::CompositeAlphaFlagsKHR::INHERIT
                }
            }
            bevy_window::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
            bevy_window::CompositeAlphaMode::PreMultiplied => {
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
            }
            bevy_window::CompositeAlphaMode::PostMultiplied => {
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
            }
            bevy_window::CompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
        },
        present_mode: match window.present_mode {
            bevy_window::PresentMode::AutoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                    vk::PresentModeKHR::FIFO_RELAXED
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy_window::PresentMode::AutoNoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                    vk::PresentModeKHR::IMMEDIATE
                } else if supported_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy_window::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            bevy_window::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            bevy_window::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            bevy_window::PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        },
        clipped: config.clipped,
        queue_family_index: config.queue_family_index,
    }
}

/// Component providing access to the current swapchain image for a window.
///
/// This component is automatically added to window entities with swapchains.
/// It contains the currently acquired swapchain image (if any) and its
/// synchronization state.
///
/// # Availability
///
/// The current image is only available between acquisition and presentation.
/// Systems should check if an image is available before using it.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::swapchain::{SwapchainImage, SwapchainSet};
///
/// fn render_to_swapchain(
///     query: Query<&SwapchainImage, With<bevy_window::PrimaryWindow>>,
/// ) {
///     let swapchain_image = query.single().unwrap();
///
///     // Image may be None if acquisition failed this frame
///     if let Some(image) = swapchain_image.current_image() {
///         // Render to image...
///     }
/// }
///
/// // Must be in SwapchainSet to access the image
/// App::new().add_systems(PostUpdate, render_to_swapchain.in_set(SwapchainSet));
/// ```
#[derive(Component)]
pub struct SwapchainImage {
    /// The acquired swapchain image for this frame, if acquired and available.
    inner: Option<GPUMutex<pumicite::swapchain::SwapchainImageInner>>,
    /// Current resource state for synchronization.
    pub state: ResourceState,
}

impl SwapchainImage {
    /// Returns the swapchain image for the current frame. May be None if swapchain image acquire
    /// was unsuccessful for the current frame, or if this function was called prior to swapchain
    /// creation.
    ///
    /// If this function returns None unexpectedly, check that your system was added to the
    /// [`SwapchainSet`] system set.
    pub fn current_image(&self) -> Option<&GPUMutex<pumicite::swapchain::SwapchainImageInner>> {
        self.inner.as_ref()
    }
}

/// Acquires the next image from the swapchain by calling `vkAcquireNextImageKHR`.
/// Generic parameter `Filter` is used to uniquely specify the swapchain to acquire from.
/// For example, `With<PrimaryWindow>` will only acquire the next image from the swapchain
/// associated with the primary window.
pub fn acquire_swapchain_image<Filter: QueryFilter>(
    mut query: Query<
        (
            Entity,
            &mut Swapchain, // Need mutable reference to swapchain to call acquire_next_image
            &mut SwapchainImage,
        ),
        Filter,
    >,
    mut suboptimal_messages: MessageWriter<SuboptimalMessage>,
) {
    use bevy_ecs::query::QuerySingleError;
    let (entity, mut swapchain, mut swapchain_image) = match query.single_mut() {
        Ok(item) => item,
        Err(QuerySingleError::NoEntities(_str)) => {
            return;
        }
        Err(QuerySingleError::MultipleEntities(str)) => {
            panic!("{}", str)
        }
    };
    assert!(
        swapchain_image.inner.is_none(),
        "This swapchain image has already been acquired!"
    );
    let new_image = match swapchain.acquire() {
        Ok((image, suboptimal)) => {
            if suboptimal {
                tracing::warn!("Swapchain acquire received suboptimal: {:?}", entity);
                suboptimal_messages.write(SuboptimalMessage { window: entity });
            }
            image
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            tracing::warn!("Swapchain acquire out-of-date: {:?}", entity);
            suboptimal_messages.write(SuboptimalMessage { window: entity });
            return;
        }
        Err(e) => {
            panic!("Error acquiring swapchain image: {:?}", e);
        }
    };

    swapchain_image.inner = Some(new_image);
    swapchain_image.state = ResourceState::default();
    swapchain_image.state.write.stage = vk::PipelineStageFlags2::ALL_COMMANDS;
}

/// Present the swapchain image by calling `vkQueuePresentKHR`.
/// Generic parameter `Filter` is used to uniquely specify the swapchain to acquire from.
/// For example, `With<PrimaryWindow>` will only acquire the next image from the swapchain
/// associated with the primary window.
pub fn present<Filter: QueryFilter>(
    mut queue: Queue<RenderQueue>, // TODO: this assumes that the render queue is capable of present
    mut query: Query<(Entity, &mut Swapchain, &mut SwapchainImage)>,
    mut suboptimal_messages: MessageWriter<SuboptimalMessage>,
) {
    use bevy_ecs::query::QuerySingleError;
    let (entity, mut swapchain, mut swapchain_image) = match query.single_mut() {
        Ok(item) => item,
        Err(QuerySingleError::NoEntities(_str)) => {
            return;
        }
        Err(QuerySingleError::MultipleEntities(str)) => {
            panic!("{}", str)
        }
    };
    let image_state = std::mem::take(&mut swapchain_image.state);
    let Some(swapchain_image) = swapchain_image.inner.take() else {
        tracing::warn!("Missing swapchain image; present skipped");
        return;
    };
    match swapchain.present(swapchain_image, image_state, &mut queue) {
        Ok(_) => {}
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            tracing::warn!("Swapchain present out-of-date: {:?}", entity);
            suboptimal_messages.write(SuboptimalMessage { window: entity });
        }
        Err(vk::Result::SUBOPTIMAL_KHR) => {
            tracing::warn!("Swapchain present suboptimal: {:?}", entity);
            suboptimal_messages.write(SuboptimalMessage { window: entity });
        }
        Err(e) => {
            panic!("Error presenting swapchain image: {:?}", e);
        }
    }
}
