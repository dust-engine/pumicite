/*!
Vulkan swapchain management for presenting rendered images to windows.

A swapchain is a queue of images that can be presented to a surface. This module
handles swapchain creation, image acquisition, and presentation for Bevy windows.

# Swapchain Image Lifecycle

1. **Creation**: When the primary window gets a surface, the [`Swapchain`] resource is
   automatically created with default or custom [`SwapchainConfig`]. There's one swapchain
   per world.

2. **Acquisition**: Before rendering each frame, an image is acquired from the swapchain
   via [`acquire_swapchain_image`].

3. **Rendering**: Systems in [`SwapchainSet`] get the current image from the
   [`Swapchain`] resource with [`Swapchain::current_image`], and render to it.

4. **Presentation**: After [`SwapchainSet`], the image is transitioned into
   [`vk::ImageLayout::PRESENT_SRC_KHR`]. The submission that [`SwapchainSet`] is part of
   waits for the image to be acquired, and signals the semaphore that presentation waits
   on. Then the image is presented via [`present`].

5. **Recreation**: The swapchain is automatically recreated when the window is resized
   or becomes suboptimal.

# Configuration

Add a [`SwapchainConfig`] component to the primary window entity to customize swapchain behavior:

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
Every system that needs to access the swapchain image stored in the [`Swapchain`] resource needs to be ordered
to run in set [`SwapchainSet`]. This ensures your system runs after image acquisition and before presentation, and
that its commands are part of the one submission that waits for the image to be acquired.
```no_run
# use bevy::prelude::*;
# fn my_render_system(){}
use bevy_pumicite::swapchain::SwapchainSet;
App::new().add_systems(PostUpdate, my_render_system.in_set(SwapchainSet));
```
*/

use bevy_app::{App, Plugin, Startup};
use bevy_ecs::prelude::*;
use glam::Vec4;
use pumicite::ash::vk;
use pumicite::device::DeviceBuilder;
use pumicite::{
    ash::khr::surface_maintenance1::Meta as KhrSurfaceMaintenance1,
    ash::khr::swapchain_maintenance1::Meta as KhrSwapchainMaintenance1,
    ash::khr::swapchain_mutable_format::Meta as ExtSwapchainMutableFormat,
    tracking::{Access, ResourceState},
};
use pumicite::ash::khr::swapchain::Meta as KhrSwapchain;
#[cfg(feature = "bevy_window")]
use pumicite::swapchain::SwapchainColorMode;

use crate::plugin::SubmissionSetConfig;
use crate::queue::RenderQueue;
use crate::{CreateDevice, DefaultRenderSet, PumiciteApp, SubmissionState};

use super::queue::Queue;
use pumicite::{swapchain::SwapchainImageInner, sync::GPUMutex};

/// System set for rendering to the swapchain image.
///
/// Systems in this set run between swapchain image acquisition and presentation,
/// giving them access to the [`Swapchain`]'s current image. The set is part of a submission set
/// on the [`RenderQueue`], so they all record into one command buffer, and there's exactly one
/// `vkQueueSubmit` between `vkAcquireNextImageKHR` and `vkQueuePresentKHR`. That submission
/// waits for the image to be acquired, and signals the semaphore that presentation waits on.
///
/// # Ordering
///
/// - Runs **after** [`acquire_swapchain_image`] (image is ready after a call to `vkAcquireNextImageKHR`)
/// - Runs **after** [`DefaultRenderSet`], which can render offscreen images for this set to use
/// - Runs **before** the image is transitioned into [`vk::ImageLayout::PRESENT_SRC_KHR`], and
///   before [`present`] (image will be presented with a call to `vkQueuePresentKHR`)
///
/// Every system that needs to access the swapchain image stored in the [`Swapchain`] resource needs to be ordered
/// to run in set [`SwapchainSet`]. Elsewhere, [`Swapchain::current_image`] returns `None` before the image
/// is acquired (e.g. in [`DefaultRenderSet`]), or an image that the submission doesn't wait for.
/// ```no_run
///# use bevy::prelude::*;
///# fn my_render_system(){}
///  use bevy_pumicite::swapchain::SwapchainSet;
///  App::new().add_systems(PostUpdate, my_render_system.in_set(SwapchainSet));
/// ```
///
/// The set is skipped on frames where there's a swapchain but no image was acquired
/// (e.g. the window is minimized or the swapchain is out of date).
/// See [`swapchain_image_acquired`].
#[derive(SystemSet, Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct SwapchainSet;

/// The submission set that [`acquire_swapchain_image`] and [`SwapchainSet`] are in.
#[derive(SystemSet, Hash, PartialEq, Eq, Debug, Clone, Copy)]
struct SwapchainSubmissionSet;

/// Run condition for [`SwapchainSet`]. Returns false only when there's a [`Swapchain`] but no
/// image was acquired this frame, so headless apps still run the set.
pub fn swapchain_image_acquired(swapchain: Option<Res<Swapchain>>) -> bool {
    swapchain.is_none_or(|swapchain| swapchain.image.is_some())
}

/// Plugin that manages swapchain creation and presentation.
///
/// # Included in [`DefaultPlugins`](crate::DefaultPlugins)
pub struct SwapchainPlugin;
impl Plugin for SwapchainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                device_builder.enable_extension::<KhrSwapchain>().unwrap();
                device_builder.enable_extension::<KhrSurfaceMaintenance1>().ok();
                device_builder
                    .enable_extension::<ExtSwapchainMutableFormat>()
                    .unwrap();

                let has_swapchain_maintenance1 = if device_builder
                    .enable_extension::<KhrSwapchainMaintenance1>()
                    .is_ok()
                {
                    device_builder
                        .enable_feature::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>(
                            |x| &mut x.swapchain_maintenance1,
                        )
                        .is_ok()
                } else {
                    false
                };
                if !has_swapchain_maintenance1 {
                    // Without VK_KHR_swapchain_maintenance1, dropping a swapchain is techncially UB because
                    // there's no way to ensure that the vkQueuePresentKHR has finished before we're allowed
                    // to drop the semaphore used for that presentation.
                    // As a workaround, if we need to drop the present semaphore, we call vkWaitQueueIdle.
                    tracing::warn!(
                        "VK_KHR_swapchain_maintenance1 is missing from this Vulkan implementation. Without this extension, dropping a swapchain is technically undefined behavior."
                    )
                }
            })
            .before(CreateDevice),
        );

        app.add_systems(
            bevy_app::PostUpdate,
            (
                #[cfg(feature = "bevy_window")]
                extract_swapchains.after(super::surface::extract_surfaces),
                acquire_swapchain_image.in_set(SwapchainSubmissionSet),
                prepare_present
                    .in_set(SwapchainSubmissionSet)
                    .after(SwapchainSet),
                present.after(SwapchainSubmissionSet),
            )
                .chain(),
        );
        app.add_submission_set::<RenderQueue>(
            SwapchainSubmissionSet,
            SubmissionSetConfig {
                debug_color: Vec4::new(1.0, 0.0, 1.0, 1.0),
            },
        );
        app.configure_sets(
            bevy_app::PostUpdate,
            (
                SwapchainSet
                    .in_set(SwapchainSubmissionSet)
                    .after(acquire_swapchain_image)
                    .run_if(swapchain_image_acquired),
                DefaultRenderSet.before(SwapchainSubmissionSet),
            ),
        );

        app.add_message::<SuboptimalMessage>();
    }
}

/// Configuration for swapchain creation.
///
/// Add this component to the primary window entity to customize swapchain behavior.
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
#[cfg(feature = "bevy_window")]
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

    /// Surface transform to apply (rotation, mirroring).
    ///
    /// Default: `IDENTITY`
    pub pre_transform: vk::SurfaceTransformFlagsKHR,

    /// Whether pixels outside the visible region can be discarded.
    /// Setting to `true` may improve performance.
    ///
    /// Default: `true`
    pub clipped: bool,

    /// Swapchain color output mode. Only used when [`image_format`](Self::image_format) is `None`.
    ///
    /// See [`SwapchainColorMode`] for details on each mode.
    ///
    /// Default: `SDR { prefer_10bit: false }`
    pub color_mode: SwapchainColorMode,
}
#[cfg(feature = "bevy_window")]
impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: None,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            required_feature_flags: vk::FormatFeatureFlags::empty(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            clipped: true,
            color_mode: SwapchainColorMode::SDR8Bit,
        }
    }
}

/// Requests that the swapchain be recreated.
#[derive(Message)]
pub struct SuboptimalMessage;

#[cfg(feature = "bevy_window")]
fn recreate_swapchain(
    swapchain: &mut pumicite::swapchain::Swapchain,
    surface: &pumicite::Surface,
    window: &bevy_window::Window,
    config: Option<&SwapchainConfig>,
) {
    use pumicite::HasDevice;
    let default_config = SwapchainConfig::default();
    let config = config.unwrap_or(&default_config);
    let create_info = get_create_info(
        surface,
        swapchain.device().physical_device(),
        window,
        config,
    );
    match swapchain.recreate(&create_info) {
        Ok(()) => {}
        // The surface changed again or has a zero extent (e.g. minimized window).
        // The swapchain stays invalid, so acquire keeps requesting a recreate until this succeeds.
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            tracing::debug!("Swapchain recreate deferred: surface out of date");
        }
        Err(e) => panic!("Error recreating swapchain: {:?}", e),
    }
}

#[cfg(feature = "bevy_window")]
pub(super) fn extract_swapchains(
    mut commands: Commands,
    device: Res<pumicite::Device>,
    queue_config: Res<crate::queue::QueueConfiguration>,
    mut window_resized_messages: MessageReader<bevy_window::WindowResized>,
    mut suboptimal_messages: MessageReader<SuboptimalMessage>,
    window: Query<
        (
            Entity,
            &bevy_window::Window,
            Option<&SwapchainConfig>,
            &pumicite::Surface,
        ),
        With<bevy_window::PrimaryWindow>,
    >,
    swapchain: Option<ResMut<Swapchain>>,
    #[cfg(target_vendor = "apple")] _marker: Option<NonSend<bevy_ecs::system::NonSendMarker>>,
) {
    let Ok((entity, window, config, surface)) = window.single() else {
        if swapchain.is_some() {
            // The primary window was closed.
            commands.remove_resource::<Swapchain>();
        }
        return;
    };
    let Some(mut swapchain) = swapchain else {
        // `present` presents on the render queue.
        let render_queue_family = queue_config
            .queue_info::<RenderQueue>()
            .unwrap()
            .family_index;
        assert!(
            device
                .physical_device()
                .supports_surface(surface, render_queue_family)
                .unwrap(),
            "The render queue (family {render_queue_family}) can't present to the primary window's surface. The selected GPU may be unable to present to this display."
        );
        let default_config = SwapchainConfig::default();
        let swapchain_config = config.unwrap_or(&default_config);
        let create_info =
            get_create_info(surface, device.physical_device(), window, swapchain_config);
        tracing::info!(
            "Swapchain created with {:?} and {:?}",
            create_info.image_format,
            create_info.image_color_space
        );
        let new_swapchain = pumicite::swapchain::Swapchain::create(
            device.clone(),
            surface.clone(),
            &create_info,
        )
        .unwrap();
        commands.insert_resource(Swapchain::new(new_swapchain));
        return;
    };

    // https://gist.github.com/nanokatze/bb03a486571e13a7b6a8709368bd87cf#file-handling-window-resize-md
    let resized = window_resized_messages
        .read()
        .filter(|message| message.window == entity)
        .count()
        > 0
        && glam::UVec2::new(window.physical_width(), window.physical_height())
            != swapchain.extent();
    let suboptimal = suboptimal_messages.read().count() > 0;
    if resized || suboptimal {
        recreate_swapchain(&mut swapchain.inner, surface, window, config);
    }
}

#[cfg(feature = "bevy_window")]
fn get_create_info(
    surface: &pumicite::Surface,
    pdevice: &pumicite::physical_device::PhysicalDevice,
    window: &bevy_window::Window,
    config: &SwapchainConfig,
) -> pumicite::swapchain::SwapchainCreateInfo {
    use glam::UVec2;
    let surface_capabilities = pdevice.get_surface_capabilities(surface).unwrap();
    let supported_present_modes = pdevice.get_surface_present_modes(surface).unwrap();
    let image_format = config.image_format.unwrap_or_else(|| {
        pumicite::swapchain::get_surface_preferred_format(
            surface,
            pdevice,
            config.required_feature_flags,
            &config.color_mode,
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
    pumicite::swapchain::SwapchainCreateInfo {
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
        present_scaling: vk::PresentScalingFlagsKHR::empty(),
        present_gravity: [vk::PresentGravityFlagsKHR::empty(); 2],
    }
}

/// The primary window's swapchain, and the image acquired from it for the current frame.
///
/// With the `bevy_window` feature, it's inserted once the primary window has a surface,
/// recreated when the window is resized, and removed when the window is closed. Without it,
/// insert one made with [`Swapchain::new`], and [`recreate`](Self::recreate) it on
/// [`SuboptimalMessage`].
///
/// # Availability
///
/// The current image is only available between acquisition and presentation, in
/// [`SwapchainSet`]. Systems should check if an image is available before using it.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_pumicite::SubmissionState;
/// use bevy_pumicite::swapchain::{Swapchain, SwapchainSet};
///
/// fn render_to_swapchain(mut swapchain: ResMut<Swapchain>, mut state: SubmissionState) {
///     // Image may be None if acquisition failed this frame
///     if let Some(image) = swapchain.current_image() {
///         // Render to image...
///     }
/// }
///
/// // Must be in SwapchainSet to access the image
/// App::new().add_systems(PostUpdate, render_to_swapchain.in_set(SwapchainSet));
/// ```
#[derive(Resource)]
pub struct Swapchain {
    inner: pumicite::swapchain::Swapchain,
    /// The acquired image, from [`acquire_swapchain_image`] until [`present`].
    ///
    /// A new `GPUMutex` is made for every acquired image. Locking it doesn't wait on anything,
    /// as the [`SwapchainSet`] submission waits for the image to be acquired instead.
    image: Option<GPUMutex<SwapchainImageInner>>,
    /// The current image's resource state.
    state: ResourceState,
}

impl Swapchain {
    /// The resource for `swapchain`, with no image acquired.
    pub fn new(inner: pumicite::swapchain::Swapchain) -> Self {
        Self {
            inner,
            image: None,
            state: ResourceState::default(),
        }
    }

    /// The extent of the swapchain's images. Unlike the current image's, it's known before
    /// the image is acquired, e.g. to size the images that [`DefaultRenderSet`] renders.
    pub fn extent(&self) -> glam::UVec2 {
        self.inner.extent()
    }

    /// Recreates the swapchain with `info`, e.g. after a [`SuboptimalMessage`]. Call it
    /// outside of [`acquire_swapchain_image`]..[`present`], while no image is acquired.
    pub fn recreate(
        &mut self,
        info: &pumicite::swapchain::SwapchainCreateInfo,
    ) -> pumicite::ash::VkResult<()> {
        self.inner.recreate(info)
    }

    /// Returns the swapchain image for the current frame. May be None if swapchain image
    /// acquire was unsuccessful for the current frame.
    ///
    /// The [`SwapchainSet`] submission waits for the image to be acquired before any of its
    /// commands start. The image is transitioned into [`vk::ImageLayout::PRESENT_SRC_KHR`]
    /// after [`SwapchainSet`].
    ///
    /// Only call this from systems in [`SwapchainSet`]. Systems that run before
    /// [`acquire_swapchain_image`], such as those in [`DefaultRenderSet`], get `None`. Systems in
    /// other submission sets that run between acquire and present get the image, but their
    /// submission doesn't wait for it to be acquired.
    pub fn current_image(&self) -> Option<&GPUMutex<SwapchainImageInner>> {
        self.image.as_ref()
    }

    /// The current image's resource state, for the barriers of the work using it.
    ///
    /// It starts with a write at [`vk::PipelineStageFlags2::ALL_COMMANDS`], so that the first
    /// barrier on the image waits for it to be acquired. Don't reset it.
    pub fn state_mut(&mut self) -> &mut ResourceState {
        &mut self.state
    }
}

/// Acquires the next image from the swapchain by calling `vkAcquireNextImageKHR`.
///
/// Runs in the submission set that [`SwapchainSet`] is in, and makes that submission wait for
/// the acquire and signal the semaphore that [`present`] waits on.
pub fn acquire_swapchain_image(
    mut submission: SubmissionState,
    swapchain: Option<ResMut<Swapchain>>,
    mut suboptimal_messages: MessageWriter<SuboptimalMessage>,
) {
    let Some(swapchain) = swapchain else {
        return;
    };
    let swapchain = swapchain.into_inner();
    assert!(
        swapchain.image.is_none(),
        "This swapchain image has already been acquired!"
    );
    if !swapchain.inner.is_valid() {
        // Already known to be out of date (e.g. window minimized); retry the recreate quietly.
        suboptimal_messages.write(SuboptimalMessage);
        return;
    }
    let new_image = match swapchain.inner.acquire() {
        Ok((image, suboptimal)) => {
            if suboptimal {
                tracing::warn!("Swapchain acquire received suboptimal");
                suboptimal_messages.write(SuboptimalMessage);
            }
            image
        }
        Err(e) if e == vk::Result::ERROR_OUT_OF_DATE_KHR => {
            tracing::warn!("Swapchain acquire out-of-date");
            suboptimal_messages.write(SuboptimalMessage);
            return;
        }
        Err(e) => {
            panic!("Error acquiring swapchain image: {:?}", e);
        }
    };

    // Safety: the acquire signals the acquire semaphore, and only this submission waits on it.
    // The present semaphore is unsignaled: the image's previous present waited on it, and was
    // done with the image before the image could be acquired again. `present` presents the
    // image this frame, and the image destroys both semaphores only after its last present
    // finishes.
    unsafe {
        submission
            .set_binary_semaphores(new_image.acquire_semaphore(), new_image.present_semaphore());
    }
    swapchain.image = Some(GPUMutex::from_boxed(new_image));
    swapchain.state = ResourceState::default();
    // The submission waits for the acquire at ALL_COMMANDS. The first barrier on the image
    // must include that stage in its source stages, so that its layout transition happens
    // after the acquire.
    swapchain.state.write.stage = vk::PipelineStageFlags2::ALL_COMMANDS;
}

/// Transitions the swapchain image into [`vk::ImageLayout::PRESENT_SRC_KHR`], after the systems
/// in [`SwapchainSet`] are done with it.
fn prepare_present(mut submission: SubmissionState, swapchain: Option<ResMut<Swapchain>>) {
    let Some(swapchain) = swapchain else {
        return;
    };
    let swapchain = swapchain.into_inner();
    submission.record(|encoder| {
        let Some(image) = swapchain.current_image() else {
            return;
        };
        let image = encoder.lock(image, vk::PipelineStageFlags2::ALL_COMMANDS);
        // ALL_COMMANDS orders the layout transition before the present semaphore's signal,
        // which is also at ALL_COMMANDS.
        encoder.use_image_resource(
            image,
            swapchain.state_mut(),
            Access::ALL_COMMANDS,
            vk::ImageLayout::PRESENT_SRC_KHR,
            0..1,
            0..image.array_layer_count(),
            false,
        );
        encoder.emit_barriers();
    });
}

/// Present the swapchain image by calling `vkQueuePresentKHR`.
///
/// Runs after the submission that [`SwapchainSet`] is part of, which signals the semaphore that
/// presentation waits on.
pub fn present(
    // The SwapchainSet submission is on this queue too, so the swapchain images never change
    // queue family. extract_swapchains checks that this family can present to the surface.
    mut queue: Queue<RenderQueue>,
    swapchain: Option<ResMut<Swapchain>>,
    mut suboptimal_messages: MessageWriter<SuboptimalMessage>,
) {
    let Some(swapchain) = swapchain else {
        return;
    };
    let swapchain = swapchain.into_inner();
    let Some(swapchain_image) = swapchain.image.take() else {
        // Expected while the window is minimized or the swapchain is out of date.
        tracing::debug!("Missing swapchain image; present skipped");
        return;
    };
    // Safety: the host only uses the image's handles from here on. `vkQueuePresentKHR` waits
    // for the semaphore that the submission signals once it's done with the image, and the
    // next submission to use the image waits for it to be acquired again.
    let (swapchain_image, _, _) = unsafe { swapchain_image.into_inner() };
    queue.begin_label(c"Swapchain Present", Vec4::new(0.0, 0.0, 0.0, 1.0));
    match swapchain.inner.present(swapchain_image, &mut queue) {
        Ok(_) => {}
        Err(e) if e == vk::Result::ERROR_OUT_OF_DATE_KHR => {
            tracing::warn!("Swapchain present out-of-date");
            suboptimal_messages.write(SuboptimalMessage);
        }
        Err(e) if e == vk::Result::SUBOPTIMAL_KHR => {
            tracing::warn!("Swapchain present suboptimal");
            suboptimal_messages.write(SuboptimalMessage);
        }
        Err(e) => {
            panic!("Error presenting swapchain image: {:?}", e);
        }
    }
    queue.end_label();
}
