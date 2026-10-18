//! Device fault reporting.
//!
//! After a device loss, [`Device::fault_report`] asks the driver what went wrong
//! through `VK_EXT_device_fault`: the faulting GPU addresses (page faults,
//! instruction pointers) and vendor-specific fault codes.
//!
//! Calls that can observe a device loss (queue submission, presentation, and
//! GPU waits) return a [`DeviceResult`]. Its [`DeviceError`] fetches the fault
//! report when formatted, so unwrapping a `DeviceResult` that failed with
//! `ERROR_DEVICE_LOST` puts the report in the panic message.
//!
//! # Requirements
//!
//! Requires the `VK_EXT_device_fault` device extension and its `deviceFault`
//! feature to be enabled on the device. Without them, a device loss is reported
//! as usual, without fault details.

use std::fmt;

use ash::{VkResult, ext::device_fault::Meta as DeviceFaultExt, vk};

use crate::Device;

/// Result of a call that can observe a device loss.
pub type DeviceResult<T> = Result<T, DeviceError>;

/// A [`vk::Result`] error from a call that can observe a device loss.
///
/// For `ERROR_DEVICE_LOST` it keeps the device, and its [`Debug`] and
/// [`Display`](fmt::Display) output include the driver's
/// [`DeviceFaultReport`]. Any other error formats as the plain [`vk::Result`].
#[derive(Clone)]
pub struct DeviceError {
    pub result: vk::Result,
    /// Only set for `ERROR_DEVICE_LOST`.
    device: Option<Device>,
}

impl DeviceError {
    pub fn new(result: vk::Result, device: &Device) -> Self {
        Self {
            result,
            device: (result == vk::Result::ERROR_DEVICE_LOST).then(|| device.clone()),
        }
    }

    /// Queries the driver for fault details. Returns `None` unless this is an
    /// `ERROR_DEVICE_LOST` from a call made through a [`Device`].
    pub fn fault_report(&self) -> Option<VkResult<DeviceFaultReport>> {
        self.device.as_ref().map(Device::fault_report)
    }
}

/// An error without a device to query, e.g. from a call that cannot lose the device.
impl From<vk::Result> for DeviceError {
    fn from(result: vk::Result) -> Self {
        Self {
            result,
            device: None,
        }
    }
}

impl From<DeviceError> for vk::Result {
    fn from(error: DeviceError) -> Self {
        error.result
    }
}

impl PartialEq<vk::Result> for DeviceError {
    fn eq(&self, other: &vk::Result) -> bool {
        self.result == *other
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.result)?;
        match self.fault_report() {
            None => Ok(()),
            Some(Ok(report)) if report.is_empty() => {
                write!(f, " (the driver reported no fault details)")
            }
            Some(Ok(report)) => write!(f, "\n{report}"),
            Some(Err(err)) => write!(f, " (fault report unavailable: {err:?})"),
        }
    }
}

impl fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for DeviceError {}

/// Attaches a device to a [`VkResult`], turning it into a [`DeviceResult`].
pub(crate) trait ResultWithDevice<T> {
    fn with_device(self, device: &Device) -> DeviceResult<T>;
}

impl<T> ResultWithDevice<T> for VkResult<T> {
    fn with_device(self, device: &Device) -> DeviceResult<T> {
        self.map_err(|result| DeviceError::new(result, device))
    }
}

/// A GPU address range the driver associated with a device fault.
#[derive(Debug, Clone, Copy)]
pub struct DeviceFaultAddress {
    pub address_type: vk::DeviceFaultAddressTypeEXT,
    /// Lowest address the fault could have occurred at.
    pub lower_bound: vk::DeviceAddress,
    /// Highest address the fault could have occurred at.
    pub upper_bound: vk::DeviceAddress,
}

/// A vendor-specific fault record.
#[derive(Debug, Clone)]
pub struct DeviceFaultVendorInfo {
    pub description: String,
    pub code: u64,
    pub data: u64,
}

/// Fault information reported by the driver after a device loss.
#[derive(Debug, Clone)]
pub struct DeviceFaultReport {
    pub description: String,
    pub addresses: Vec<DeviceFaultAddress>,
    pub vendor_infos: Vec<DeviceFaultVendorInfo>,
    /// Size of the vendor binary crash dump the driver has available. The dump
    /// itself is not retrieved.
    pub vendor_binary_size: vk::DeviceSize,
}

impl DeviceFaultReport {
    /// Returns true if the driver reported nothing beyond an empty description.
    pub fn is_empty(&self) -> bool {
        self.description.is_empty() && self.addresses.is_empty() && self.vendor_infos.is_empty()
    }
}

impl fmt::Display for DeviceFaultReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device fault: {:?}", self.description)?;
        for address in &self.addresses {
            write!(
                f,
                "\n  {:?} at 0x{:x}..=0x{:x}",
                address.address_type, address.lower_bound, address.upper_bound
            )?;
        }
        for vendor in &self.vendor_infos {
            write!(
                f,
                "\n  vendor {:?}: code=0x{:x} data=0x{:x}",
                vendor.description, vendor.code, vendor.data
            )?;
        }
        if self.vendor_binary_size > 0 {
            write!(
                f,
                "\n  vendor binary crash dump available ({} bytes)",
                self.vendor_binary_size
            )?;
        }
        Ok(())
    }
}

impl Device {
    /// Queries `VK_EXT_device_fault` for information about a device loss.
    ///
    /// Returns `ERROR_EXTENSION_NOT_PRESENT` if the extension is not enabled and
    /// `ERROR_FEATURE_NOT_PRESENT` if the `deviceFault` feature is not enabled.
    /// Only meaningful after an operation returned `ERROR_DEVICE_LOST`.
    pub fn fault_report(&self) -> VkResult<DeviceFaultReport> {
        let ext = self
            .get_extension::<DeviceFaultExt>()
            .map_err(|_| vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
        let feature_enabled = self
            .feature::<vk::PhysicalDeviceFaultFeaturesEXT>()
            .is_some_and(|f| f.device_fault == vk::TRUE);
        if !feature_enabled {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let get_fault_info = ext.fp().get_device_fault_info_ext;

        let mut counts = vk::DeviceFaultCountsEXT::default();
        unsafe { get_fault_info(self.handle(), &mut counts, std::ptr::null_mut()) }.result()?;

        let mut addresses =
            vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendor_infos =
            vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        let vendor_binary_size = counts.vendor_binary_size;
        // The vendor binary dump is not retrieved.
        counts.vendor_binary_size = 0;
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addresses.as_mut_ptr(),
            p_vendor_infos: vendor_infos.as_mut_ptr(),
            ..Default::default()
        };
        let result = unsafe { get_fault_info(self.handle(), &mut counts, &mut info) };
        if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
            return Err(result);
        }
        addresses.truncate(counts.address_info_count as usize);
        vendor_infos.truncate(counts.vendor_info_count as usize);

        Ok(DeviceFaultReport {
            description: info
                .description_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            addresses: addresses
                .iter()
                .map(|a| {
                    // addressPrecision is a power of two; the fault lies within
                    // the aligned block containing reportedAddress.
                    let mask = a.address_precision.max(1) - 1;
                    DeviceFaultAddress {
                        address_type: a.address_type,
                        lower_bound: a.reported_address & !mask,
                        upper_bound: a.reported_address | mask,
                    }
                })
                .collect(),
            vendor_infos: vendor_infos
                .iter()
                .map(|v| DeviceFaultVendorInfo {
                    description: v
                        .description_as_c_str()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    code: v.vendor_fault_code,
                    data: v.vendor_fault_data,
                })
                .collect(),
            vendor_binary_size,
        })
    }
}
