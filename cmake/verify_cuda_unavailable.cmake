if(NOT DEFINED NBODY_HEADLESS OR NOT DEFINED NBODY_OUTPUT_ROOT)
    message(FATAL_ERROR "NBODY_HEADLESS and NBODY_OUTPUT_ROOT are required")
endif()

foreach(solver direct barnes-hut)
    execute_process(
        COMMAND "${NBODY_HEADLESS}"
                --backend cuda
                --solver "${solver}"
                --steps 1
                --sample-every 1
                --output "${NBODY_OUTPUT_ROOT}_${solver}.csv"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE standard_output
        ERROR_VARIABLE standard_error
    )

    if(result EQUAL 0)
        message(FATAL_ERROR
            "CPU-only headless runner unexpectedly accepted CUDA ${solver}")
    endif()

    set(expected
        "CUDA backend is unavailable in this build; configure with NBODY_ENABLE_CUDA=ON")
    set(combined "${standard_output}${standard_error}")
    string(FIND "${combined}" "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "CPU-only CUDA ${solver} rejection used the wrong diagnostic. Output:\n${combined}")
    endif()
endforeach()
