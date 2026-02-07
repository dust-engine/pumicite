.PHONY: default
default: egui.spv

egui.spv: egui.slang
	slangc -O3 -target spirv -o egui.spv egui.slang
