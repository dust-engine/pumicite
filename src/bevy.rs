use std::{ops::Deref, sync::Arc};

use bevy_asset::Asset;
use bevy_ecs::{
    component::{Component, Immutable, Mutable, StorageType},
    resource::Resource,
    world::FromWorld,
};
use bevy_reflect::TypePath;

use crate::Device;

impl Resource for crate::Device {}
impl Resource for crate::Instance {}
impl Resource for crate::Allocator {}
impl Resource for crate::physical_device::PhysicalDevice {}
impl Resource for crate::device::DeviceBuilder {}
impl Resource for crate::instance::InstanceBuilder {}

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

impl Resource for crate::debug::DebugUtilsMessenger {}

impl Component for crate::Surface {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Immutable;
}

impl Component for crate::swapchain::Swapchain {
    const STORAGE_TYPE: StorageType = StorageType::SparseSet;

    type Mutability = Mutable;
}
