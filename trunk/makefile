
SHELL = /bin/sh

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${PETSC_DIR}/conf/variables

include ${DENDRO_DIR}/makeVariables

CEXT = C
CFLAGS = -DPETSC_USE_LOG -O3 

#-D__USE_MG_INIT_TYPE2__
#-D__USE_64_BIT_INT__

#-g -O0

include ./makefileCore

