import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


def test_energy_breakdown():
    """
    lightning_doc
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                MAX_UTILIZATION=False
                # Albireo authors kept the frequency constant for this specific
                # table.
                # GLOBAL_CYCLE_SECONDS=0.2e-9,
            ),
        )
        for s in ["conservative"]
    )

    def w2pj(*args):  # * 97GHz * 1e12 J->pJ
        return [y * 0.01030927e-9 * 1e12 for y in args]

    # results.consolidate_energy()
    results.add_compare_ref_energy("laser", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("photodetector", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("individual_modulator_placeholder", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("WMUs", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("adc", w2pj(0.075))
    results.add_compare_ref_energy("input_dac", w2pj(0.077))
    results.add_compare_ref_energy("weight_dacs", w2pj(0.077))
    results.add_compare_ref_energy("memory_controller", w2pj(0.0186))
    results.add_compare_ref_energy("packet_io_input", w2pj(0.009))
    results.add_compare_ref_energy("packet_io_output", w2pj(0.009))

    return results


def test_area_breakdown():
    """
    lightning_doc
    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))

    total_area = 2095.787 * 1000000  # um^2
    expected_area = {
        "Packet I/O": 0.00009876957916047766 * total_area * 2, # x2 because we have both input and output packer IO
        "Memory Controller": 0.03549979077072240642 * total_area,
        "DAC": 0.16604740844370157845 * total_area,
        "ADC": 0.00664189633774806313 * total_area,
        "input_modulator": 0.02862886352477613421 * total_area,
        "weight_modulator": 0.68709272459462722118 * total_area,
        "Photodetector": 0.00000036644945311713 * total_area,
        "Laser": 0.00000477147725412935 * total_area
    }

    results.consolidate_area(["packet_io_input", "packet_io_output"], "Packet I/O")
    results.add_compare_ref_area("Packet I/O", [expected_area["Packet I/O"]])
    results.consolidate_area(["memory_controller"], "Memory Controller")
    results.add_compare_ref_area("Memory Controller", [expected_area["Memory Controller"]])
    results.consolidate_area(["weight_dacs", "input_dac"], "DAC")
    results.add_compare_ref_area("DAC", [expected_area["DAC"]])
    results.consolidate_area(["adc"], "ADC")
    results.add_compare_ref_area("ADC", [expected_area["ADC"]])
    results.consolidate_area(["WMUs"], "weight_modulator")
    results.add_compare_ref_area("weight_modulator", [expected_area["weight_modulator"]])
    results.consolidate_area(["individual_modulator_placeholder"], "input_modulator")
    results.add_compare_ref_area("input_modulator", [expected_area["input_modulator"]])
    results.consolidate_area(["photodetector"], "Photodetector")
    results.add_compare_ref_area("Photodetector", [expected_area["Photodetector"]])
    results.consolidate_area(["laser"], "Laser")
    results.add_compare_ref_area("Laser", [expected_area["Laser"]])

    results.clear_zero_areas()
    return results


def test_explore_architectures(dnn_name: str):
    """
    lightning_doc
    """
    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    layer_paths = [l for l in layer_paths if "From einsum" not in open(l, "r").read()]

    def callfunc(spec):  # Speed up the test by reducing the victory condition
        spec.mapper.victory_condition = 10

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(
                SCALING=f'"{s}"',
                N_COLUMNS=x,
                N_ROWS=y,
                N_STAR_COUPLED_GROUPS_OF_ROWS=z,
                GLB_DEPTH_SCALE=g,
                N_PLCU=n_plcu,
                N_PLCG=n_plcg,
            ),
            system="ws_dummy_buffer_one_macro",
            callfunc=callfunc,
        )
        for l in layer_paths
        for s in ["aggressive"]
        for x, y, z in [(5, 3, 3), (7, 3, 1)]
        for g in [1]
        for n_plcu in [3, 9, 15]
        for n_plcg in [9, 27, 45]
    )

    results.consolidate_energy(
        ["weight_mach_zehnder_modulator", "weight_dac", "weight_cache"],
        "Weight Processing",
    )
    results.consolidate_energy(
        ["input_mach_zehnder_modulator", "input_dac", "input_MRR"],
        "Input Processing",
    )
    results.consolidate_energy(["adc", "output_regs", "TIA"], "Output Processing")
    results.consolidate_energy(["laser", "MRR", "global_buffer"], "Other")
    results.clear_zero_energies()
    return results.aggregate_by("N_COLUMNS", "N_PLCU", "N_PLCG")


def test_full_dnn(dnn_name: str, batch_sizes: list, num_parallel_wavelengths: list, num_parallel_batches: list, num_parallel_weights: list):
    """
    lightning_doc
    """
    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    def callfunc(spec):  # Speed up the test by reducing the victory condition
        spec.mapper.victory_condition = 10

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(
                BATCH_SIZE=n,
                NUM_WAVELENGTHS=w,
                NUM_PARALLEL_WEIGHTS=pw,
                PARALLEL_BATCH_SIZE=b,
                SCALING=f'"{s}"',
            ),
            system="ws_dummy_buffer_one_macro",
            callfunc=callfunc,
        )
        for s in ["conservative"]
        for n in batch_sizes
        for l in layer_paths
        for w in num_parallel_wavelengths
        for pw in num_parallel_weights
        for b in num_parallel_batches
    )
    return results

if __name__ == "__main__":
    test_energy_breakdown()
    test_area_breakdown()
    test_full_dnn("alexnet")
    test_full_dnn("vgg16")
    test_explore_architectures("resnet18")
    test_explore_main_memory("resnet18")
