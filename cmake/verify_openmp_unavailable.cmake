if(NOT DEFINED NBODY_HEADLESS OR NOT DEFINED NBODY_OUTPUT)
    message(FATAL_ERROR "NBODY_HEADLESS and NBODY_OUTPUT are required")
endif()

execute_process(
    COMMAND "${NBODY_HEADLESS}"
            --backend openmp
            --solver barnes-hut
            --threads 2
            --steps 1
            --sample-every 1
            --output "${NBODY_OUTPUT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE standard_output
    ERROR_VARIABLE standard_error
)

if(result EQUAL 0)
    message(FATAL_ERROR "CPU-only headless runner unexpectedly accepted OpenMP Barnes-Hut")
endif()

set(expected "OpenMP backend is unavailable in this build; configure with NBODY_ENABLE_OPENMP=ON")
set(combined "${standard_output}${standard_error}")
string(FIND "${combined}" "${expected}" position)
if(position EQUAL -1)
    message(FATAL_ERROR
        "CPU-only rejection used the wrong diagnostic. Output:\n${combined}")
endif()
