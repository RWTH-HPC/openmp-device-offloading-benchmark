# tool macros
CC := hipcc
INCLUDE_ALLOC ?= 1
CCFLAGS ?= -O3 -std=c++17 -fopenmp --offload-arch=gfx90a -DINCLUDE_ALLOC=${INCLUDE_ALLOC}
DBGFLAGS := -g
CCOBJFLAGS := $(CCFLAGS) -c
TARGET_EXT ?= default

# path macros
BIN_PATH := bin/${TARGET_EXT}
OBJ_PATH := obj/${TARGET_EXT}
DBG_PATH := debug/${TARGET_EXT}
SRC_PATH := .

# compile macros
TARGET_NAME := hip_bandwidth_${TARGET_EXT}
TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cc)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# clean files list
DISTCLEAN_LIST := $(OBJ) \
                  $(OBJ_DEBUG)
CLEAN_LIST := $(TARGET) \
			  $(TARGET_DEBUG) \
			  $(DISTCLEAN_LIST)

# default rule
default: makedir all

# non-phony targets
$(TARGET): $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $(OBJ)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) $(DBGFLAGS) -o $@ $<

$(TARGET_DEBUG): $(OBJ_DEBUG)
	$(CC) $(CCFLAGS) $(DBGFLAGS) $(OBJ_DEBUG) -o $@

# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)

# Note: might be good to disable automatic numabalancing on system
.PHONY: run
run:
	OMP_PLACES=cores OMP_PROC_BIND=close $(TARGET)
	@#OMP_PLACES=cores OMP_PROC_BIND=close no_numa_balancing $(TARGET)
