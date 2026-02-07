pub mod format;
pub mod future;
mod idalloc;
use ash::vk::{self, TaggedStructure};
use glam::Affine3A;
pub use idalloc::IdAlloc;
use std::{collections::BTreeMap, ffi::c_void, fmt::Debug, ops::Deref, ptr::NonNull};

#[derive(Debug, Clone)]
pub enum SharingMode<T>
where
    T: Deref<Target = [u32]>,
{
    Exclusive,
    Concurrent { queue_family_indices: T },
}

impl<T: Deref<Target = [u32]>> SharingMode<T> {
    pub fn as_raw(&self) -> vk::SharingMode {
        match self {
            Self::Exclusive => vk::SharingMode::EXCLUSIVE,
            Self::Concurrent { .. } => vk::SharingMode::CONCURRENT,
        }
    }

    pub fn queue_family_indices(&self) -> &[u32] {
        match self {
            Self::Exclusive => &[],
            Self::Concurrent {
                queue_family_indices,
            } => queue_family_indices.deref(),
        }
    }
}

/// Type-erased object representing a tagged Vulkan structure.
/// It is basically a [`Box<dyn Any>`], but for types implementing [`ash::vk::TaggedStructure`].
#[repr(C)]
pub struct VkTaggedObject {
    pub s_type: vk::StructureType,
    pub p_next: *mut std::ffi::c_void,
    rest: [u8],
}
impl VkTaggedObject {
    pub fn from_ref<T: TaggedStructure<'static>>(obj: &T) -> &Self {
        let ptr = std::ptr::from_raw_parts::<Self>(
            obj as *const T as *const (),
            std::mem::size_of::<T>() - std::mem::size_of::<vk::BaseInStructure>(),
        );
        unsafe { &*ptr }
    }
    /// # Safety
    /// The provided object must be a valid instance of a Vulkan structure that starts with a `s_type` field.
    pub unsafe fn new_unchecked<T>(obj: T) -> Box<Self> {
        unsafe {
            let boxed = Box::new(obj);
            let boxed_ptr = NonNull::new_unchecked(Box::into_raw(boxed));
            let fat_ptr = NonNull::<Self>::from_raw_parts(
                boxed_ptr,
                std::mem::size_of::<T>() - std::mem::size_of::<vk::BaseInStructure>(),
            );
            Box::from_raw(fat_ptr.as_ptr())
        }
    }
    pub fn new<T: TaggedStructure<'static>>(obj: T) -> Box<Self> {
        unsafe { Self::new_unchecked(obj) }
    }
    pub fn downcast_ref<T: TaggedStructure<'static>>(&self) -> Option<&T> {
        if self.s_type == T::STRUCTURE_TYPE {
            Some(unsafe { &*(self as *const Self as *const T) })
        } else {
            None
        }
    }
    /// # Safety
    /// The caller must ensure that the structure type matches the generic type `T`.
    pub unsafe fn downcast_ref_to_type<T>(&self, ty: vk::StructureType) -> Option<&T> {
        if self.s_type == ty {
            unsafe { Some(&*(self as *const Self as *const T)) }
        } else {
            None
        }
    }
    pub fn downcast_mut<T: TaggedStructure<'static>>(&mut self) -> Option<&mut T> {
        if self.s_type == T::STRUCTURE_TYPE {
            Some(unsafe { &mut *(self as *mut Self as *mut T) })
        } else {
            None
        }
    }
    /// # Safety
    /// The caller must ensure that the structure type matches the generic type `T`.
    pub unsafe fn downcast_mut_to_type<T>(&mut self, ty: vk::StructureType) -> Option<&mut T> {
        if self.s_type == ty {
            unsafe { Some(&mut *(self as *mut Self as *mut T)) }
        } else {
            None
        }
    }
}

/// An utility struct representing an entire raw chain.
/// T is the base struct.
#[derive(Default)]
pub struct NextChainMap<T> {
    pub head: T,
    features: BTreeMap<vk::StructureType, Box<VkTaggedObject>>,
}

impl<'a, T: ash::vk::TaggedStructure<'a>> NextChainMap<T> {
    pub fn get<E: ash::vk::TaggedStructure<'static>>(&self) -> Option<&E> {
        // TODO: ensure E extends T on the trait bound once Extends<T> lands in ash.
        if E::STRUCTURE_TYPE == T::STRUCTURE_TYPE {
            unsafe {
                let ptr = &raw const self.head as *const E;
                return Some(&*ptr);
            }
        }
        let ty = E::STRUCTURE_TYPE;
        let enabled_features = self.features.get(&ty)?;
        let enabled_features =
            unsafe { enabled_features.deref().downcast_ref_to_type::<E>(ty) }.unwrap();
        Some(enabled_features)
    }

    pub fn get_mut_or_insert_with<E: ash::vk::TaggedStructure<'static>>(
        &mut self,
        insert: impl FnOnce(&mut T) -> E,
    ) -> &mut E {
        // TODO: ensure E extends T on the trait bound once Extends<T> lands in ash.
        if E::STRUCTURE_TYPE == T::STRUCTURE_TYPE {
            unsafe {
                let ptr = &raw mut self.head as *mut E;
                return &mut *ptr;
            }
        }
        let ty = E::STRUCTURE_TYPE;
        let enabled_features = self.features.entry(ty).or_insert_with(|| {
            let item = insert(&mut self.head);
            VkTaggedObject::new(item)
        });
        unsafe { enabled_features.downcast_mut_to_type::<E>(ty) }.unwrap()
    }
}
unsafe impl<'a, T: ash::vk::TaggedStructure<'a>> Send for NextChainMap<T> {}
unsafe impl<'a, T: ash::vk::TaggedStructure<'a>> Sync for NextChainMap<T> {}

impl<'a, T: ash::vk::TaggedStructure<'a>> NextChainMap<T> {
    pub fn make_chain(&mut self) {
        let head: &mut vk::BaseOutStructure =
            unsafe { &mut *(&raw mut self.head as *mut vk::BaseOutStructure) };
        // build p_next chain
        let mut last: &mut *mut c_void = unsafe { std::mem::transmute(&mut head.p_next) };
        for f in self.features.values_mut() {
            *last = f.as_mut() as *mut _ as *mut c_void;
            last = &mut f.p_next;
        }
    }
}

pub trait AsVkHandle {
    type Handle: ash::vk::Handle + Copy;
    fn vk_handle(&self) -> Self::Handle;
}
impl<T> AsVkHandle for &'_ T
where
    T: AsVkHandle,
{
    type Handle = T::Handle;

    fn vk_handle(&self) -> Self::Handle {
        T::vk_handle(self)
    }
}
impl<T> AsVkHandle for &'_ mut T
where
    T: AsVkHandle,
{
    type Handle = T::Handle;

    fn vk_handle(&self) -> Self::Handle {
        T::vk_handle(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u32);
impl Version {
    pub const V1_0: Self = Self::new(0, 1, 0, 0);
    pub const V1_1: Self = Self::new(0, 1, 1, 0);
    pub const V1_2: Self = Self::new(0, 1, 2, 0);
    pub const V1_3: Self = Self::new(0, 1, 3, 0);
    pub const V1_4: Self = Self::new(0, 1, 4, 0);

    pub const fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        let num = vk::make_api_version(variant, major, minor, patch);
        Self(num)
    }
    pub const fn major(&self) -> u32 {
        vk::api_version_major(self.0)
    }
    pub const fn minor(&self) -> u32 {
        vk::api_version_minor(self.0)
    }
    pub const fn patch(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub const fn variant(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub const fn as_raw(&self) -> u32 {
        self.0
    }
}
impl Default for Version {
    fn default() -> Self {
        Self::new(0, 0, 1, 0)
    }
}
impl Debug for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Version({}.{}.{})",
            self.major(),
            self.minor(),
            self.patch()
        ))?;
        let variant = self.variant();
        if variant != 0 {
            f.write_fmt(format_args!(" variant {variant}"))?;
        }
        Ok(())
    }
}
impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}.{}.{}",
            self.major(),
            self.minor(),
            self.patch()
        ))?;
        let variant = self.variant();
        if variant != 0 {
            f.write_fmt(format_args!(" variant {variant}"))?;
        }
        Ok(())
    }
}
impl From<Version> for String {
    fn from(value: Version) -> Self {
        format!(
            "{}.{}.{}:{}",
            value.major(),
            value.minor(),
            value.patch(),
            value.variant()
        )
    }
}

pub fn glam_to_vk_transform(affine: Affine3A) -> vk::TransformMatrixKHR {
    let x = &affine.matrix3.x_axis;
    let y = &affine.matrix3.y_axis;
    let z = &affine.matrix3.z_axis;
    let w = &affine.translation;
    vk::TransformMatrixKHR {
        // row major
        matrix: [x.x, y.x, z.x, w.x, x.y, y.y, z.y, w.y, x.z, y.z, z.z, w.z],
    }
}
