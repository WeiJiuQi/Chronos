BUILD_DIR ?= build
JOBS ?= 7

.PHONY: configure build check clean

configure:
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

check: build
	$(BUILD_DIR)/tango/tango_index --help
	$(BUILD_DIR)/tango/tango_query --help
	$(BUILD_DIR)/tango/tango_dynamic --help
	$(BUILD_DIR)/tango/tango_decay_profile --help
	$(BUILD_DIR)/tdvs_workload_builder/prepare_vector_subset --help
	$(BUILD_DIR)/tdvs_workload_builder/generate_time_metadata --help
	$(BUILD_DIR)/tdvs_workload_builder/generate_tdvs_groundtruth --help
	$(BUILD_DIR)/tdvs_workload_builder/export_mips_transforms --help
	$(BUILD_DIR)/tdvs_workload_builder/verify_dataset_alignment --help
	$(BUILD_DIR)/baselines/tdvs_ip_nsw --help
	$(BUILD_DIR)/baselines/tdvs_ip_nsw_plus --help
	$(BUILD_DIR)/baselines/tdvs_napg --help
	$(BUILD_DIR)/baselines/tdvs_mag --help
	$(BUILD_DIR)/baselines/tdvs_psp --help

clean:
	cmake -E remove_directory $(BUILD_DIR)
