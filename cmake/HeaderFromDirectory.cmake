MACRO(SUBDIRLIST result curdir)
FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
SET(dirlist "")
FOREACH(child ${children})
  IF(IS_DIRECTORY ${curdir}/${child})
    LIST(APPEND dirlist ${child})
  ENDIF()
ENDFOREACH()
SET(${result} ${dirlist})
ENDMACRO()

MACRO(GenerateHeaderFromDirectory inputdir storedir genfile)

    get_filename_component(foldername ${inputdir} NAME)
    set(headername "${foldername}.h")

    file(GLOB source_list "${inputdir}/*.hpp" "${inputdir}/*.h" "${inputdir}/*.cuh")

    set(CONTENT "#ifndef RMAGINE_${foldername}_H\n#define RMAGINE_${foldername}_H\n")

    FOREACH(header_file ${source_list})
        get_filename_component(header_name ${header_file} NAME)
        set(CONTENT "${CONTENT}\n#include \"${foldername}/${header_name}\"")
    ENDFOREACH()

    set(CONTENT "${CONTENT}\n\n#endif // RMAGINE_${foldername}_H")

    file(WRITE "${storedir}/${headername}" ${CONTENT})

    set(genfile "${storedir}/${headername}")

ENDMACRO()

MACRO(GenerateHeadersFromDirectories inputdir storedir genfiles)
SUBDIRLIST(SUBDIRS ${inputdir})

SET(filelist "")

FOREACH(subdir ${SUBDIRS})
  GenerateHeaderFromDirectory("${inputdir}/${subdir}" ${storedir} genfile)
  list(APPEND filelist ${genfile})
ENDFOREACH()

set(${genfiles} ${filelist})

ENDMACRO()