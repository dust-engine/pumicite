use std::{ops::Deref, sync::Arc};

use bevy_asset::Asset;
use bevy_ecs::{
    component::{Component, Immutable, Mutable, StorageType},
    resource::{IsResource, Resource},
    world::FromWorld,
};
use bevy_reflect::TypePath;

use crate::Device;

/// Implements [`Resource`] for a foreign type.
macro_rules! impl_resource {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Component for $ty {
                const STORAGE_TYPE: StorageType = StorageType::SparseSet;
                type Mutability = Mutable;

                fn register_required_components(
                    _component_id: bevy_ecs::component::ComponentId,
                    required_components: &mut bevy_ecs::component::RequiredComponentsRegistrator,
                ) {
                    // Check for an existing id first to avoid recursing during
                    // required-component initialization, as the derive does.
                    let resource_component_id = if let Some(id) =
                        required_components.components_registrator().component_id::<$ty>()
                    {
                        id
                    } else {
                        required_components
                            .components_registrator()
                            .register_component::<$ty>()
                    };
                    required_components.register_required::<IsResource>(move || {
                        IsResource::new(resource_component_id)
                    });
                }
            }
            impl Resource for $ty {}
        )*
    };
}

impl_resource!(
    crate::Device,
    crate::Instance,
    crate::Allocator,
    crate::physical_device::PhysicalDevice,
    crate::device::DeviceBuilder,
    crate::instance::InstanceBuilder,
);

#[derive(Clone, Asset, TypePath)]
pub struct PipelineLayout(pub Arc<crate::pipeline::PipelineLayout>);

impl Deref for PipelineLayout {
    type Target = Arc<crate::pipeline::PipelineLayout>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Asset, TypePath)]
pub struct DescriptorSetLayout(pub Arc<crate::descriptor::DescriptorSetLayout>);

impl Deref for DescriptorSetLayout {
    type Target = Arc<crate::descriptor::DescriptorSetLayout>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Resource)]
pub struct PipelineCache(Arc<crate::pipeline::PipelineCache>);
impl Deref for PipelineCache {
    type Target = Arc<crate::pipeline::PipelineCache>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl FromWorld for PipelineCache {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self(Arc::new(crate::pipeline::PipelineCache::null(
            world.resource::<Device>().clone(),
        )))
    }
}

impl_resource!(crate::debug::DebugUtilsMessenger);

impl Component for crate::Surface {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Immutable;
}

impl Component for crate::swapchain::Swapchain {
    const STORAGE_TYPE: StorageType = StorageType::SparseSet;

    type Mutability = Mutable;
}
