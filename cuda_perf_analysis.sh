#!/bin/bash

# General CUDA Performance Analysis Script
# Can be used with any CUDA project

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
EXECUTABLE="${1:-}"
BENCHMARK_ARGS="${2:---benchmark --iterations 1}"
PROFILER_ITERATIONS="${PROFILER_ITERATIONS:-1}"
USE_COLOR="${USE_COLOR:-true}"

# Color codes for output
if [ "$USE_COLOR" = "true" ] && [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Helper functions
print_header() {
    echo -e "${BOLD}${CYAN}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [EXECUTABLE] [BENCHMARK_ARGS]

OPTIONS:
    -h, --help              Show this help message
    -p, --project-dir DIR   Set project directory (default: current directory)
    -b, --build-dir DIR     Set build directory (default: build)
    -t, --build-type TYPE   Set build type (default: Release)
    -s, --skip-build        Skip the build step
    -n, --no-color          Disable colored output
    --profile-only          Only run profiling, skip benchmark
    --benchmark-only        Only run benchmark, skip profiling

ENVIRONMENT VARIABLES:
    PROJECT_DIR             Project root directory
    BUILD_DIR               Build directory name
    BUILD_TYPE              CMake build type (Debug/Release)
    PROFILER_ITERATIONS     Number of iterations for profiler (default: 1)
    USE_COLOR               Enable/disable colored output (true/false)

EXAMPLES:
    # Basic usage with auto-detection
    $0

    # Specify executable and arguments
    $0 ./build/my_cuda_app "--input data.txt --iterations 10"

    # Skip build and profile specific executable
    $0 --skip-build ./bin/cuda_program "--test"

    # Profile with custom project directory
    $0 --project-dir /path/to/project
EOF
}

# Parse command line arguments
SKIP_BUILD=false
PROFILE_ONLY=false
BENCHMARK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -p|--project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -n|--no-color)
            USE_COLOR=false
            # Reset color codes
            RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; NC=''
            shift
            ;;
        --profile-only)
            PROFILE_ONLY=true
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            shift
            ;;
        *)
            if [ -z "$EXECUTABLE" ]; then
                EXECUTABLE="$1"
            else
                BENCHMARK_ARGS="$1"
            fi
            shift
            ;;
    esac
done

# Start analysis
print_header "CUDA Performance Analysis"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# Check CUDA installation
print_header "Environment Check"

if check_command nvidia-smi; then
    print_success "NVIDIA driver found"
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | sed 's/^/  /'
else
    print_error "nvidia-smi not found - CUDA may not be properly installed"
fi

# Check for CUDA compiler
if check_command nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    print_success "CUDA compiler found (version $CUDA_VERSION)"
else
    print_warning "nvcc not found - CUDA toolkit may not be installed"
fi

# Check for profilers
PROFILER=""
if check_command nvprof; then
    PROFILER="nvprof"
    print_success "nvprof profiler available"
elif check_command ncu; then
    PROFILER="ncu"
    print_success "Nsight Compute (ncu) profiler available"
elif check_command nsys; then
    PROFILER="nsys"
    print_success "Nsight Systems (nsys) profiler available"
else
    print_warning "No CUDA profiler found (nvprof, ncu, or nsys)"
fi

echo

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Build the project if needed
if [ "$SKIP_BUILD" = false ]; then
    print_header "Building Project"
    
    # Check for build system
    if [ -f "CMakeLists.txt" ]; then
        print_info "Using CMake build system"
        cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        cmake --build "$BUILD_DIR" --parallel
    elif [ -f "Makefile" ]; then
        print_info "Using Makefile build system"
        make -j$(nproc)
    elif [ -f "build.sh" ]; then
        print_info "Using custom build script"
        ./build.sh
    else
        print_warning "No build system detected, skipping build"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Build successful!"
    else
        print_error "Build failed!"
        exit 1
    fi
    echo
fi

# Find executable if not specified
if [ -z "$EXECUTABLE" ]; then
    print_info "No executable specified, searching for CUDA binaries..."
    
    # Look for executables in common locations
    SEARCH_DIRS=("$BUILD_DIR" "bin" "." "build/bin" "build/Release" "build/Debug")
    
    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # Find files that might be CUDA executables
            while IFS= read -r -d '' file; do
                if file "$file" 2>/dev/null | grep -q "ELF.*executable"; then
                    # Check if it links to CUDA runtime
                    if ldd "$file" 2>/dev/null | grep -q "libcudart"; then
                        EXECUTABLE="$file"
                        print_success "Found CUDA executable: $EXECUTABLE"
                        break 2
                    fi
                fi
            done < <(find "$dir" -maxdepth 2 -type f -executable -print0 2>/dev/null)
        fi
    done
    
    if [ -z "$EXECUTABLE" ]; then
        print_error "No CUDA executable found. Please specify one as an argument."
        echo "Usage: $0 [executable] [arguments]"
        exit 1
    fi
fi

# Verify executable exists and is runnable
if [ ! -f "$EXECUTABLE" ]; then
    print_error "Executable not found: $EXECUTABLE"
    exit 1
fi

if [ ! -x "$EXECUTABLE" ]; then
    print_error "File is not executable: $EXECUTABLE"
    exit 1
fi

echo

# Run benchmark
if [ "$PROFILE_ONLY" = false ]; then
    print_header "Running Benchmark"
    echo "Command: $EXECUTABLE $BENCHMARK_ARGS"
    echo
    
    # Run with timing
    time $EXECUTABLE $BENCHMARK_ARGS
    
    if [ $? -ne 0 ]; then
        print_warning "Benchmark returned non-zero exit code"
    fi
    echo
fi

# Run profiling
if [ "$BENCHMARK_ONLY" = false ] && [ -n "$PROFILER" ]; then
    print_header "Profiling Analysis"
    
    case "$PROFILER" in
        nvprof)
            print_info "Using nvprof for analysis"
            echo
            
            # Create a temporary directory for profiling output
            PROF_DIR=$(mktemp -d)
            
            # Memory analysis
            echo "1. Memory Coalescing Efficiency:"
            nvprof --metrics gld_efficiency,gst_efficiency \
                   --print-gpu-summary \
                   $EXECUTABLE $BENCHMARK_ARGS 2>&1 | grep -E "Efficiency|throughput"
            
            echo
            echo "2. Occupancy Analysis:"
            nvprof --metrics achieved_occupancy \
                   --print-gpu-summary \
                   $EXECUTABLE $BENCHMARK_ARGS 2>&1 | grep -E "occupancy|Occupancy"
            
            echo
            echo "3. Memory Throughput:"
            nvprof --metrics dram_read_throughput,dram_write_throughput \
                   --print-gpu-summary \
                   $EXECUTABLE $BENCHMARK_ARGS 2>&1 | grep -E "throughput|Throughput"
            
            # Cleanup
            rm -rf "$PROF_DIR"
            ;;
            
        ncu)
            print_info "Using Nsight Compute for analysis"
            echo
            
            # Run comprehensive analysis
            ncu --target-processes all \
                --kernel-regex ".*" \
                --metrics regex:.*throughput.*,regex:.*efficiency.* \
                --print-summary per-kernel \
                $EXECUTABLE $BENCHMARK_ARGS
            ;;
            
        nsys)
            print_info "Using Nsight Systems for analysis"
            echo
            
            OUTPUT_FILE="nsys_report_$(date +%Y%m%d_%H%M%S)"
            nsys profile --stats=true --output="$OUTPUT_FILE" $EXECUTABLE $BENCHMARK_ARGS
            
            if [ -f "${OUTPUT_FILE}.nsys-rep" ]; then
                print_success "Profile saved to ${OUTPUT_FILE}.nsys-rep"
                echo "View with: nsys-ui ${OUTPUT_FILE}.nsys-rep"
            fi
            ;;
    esac
    echo
fi

# Performance guidelines
print_header "Performance Guidelines"

echo -e "${BOLD}Target Metrics:${NC}"
echo "  • Global Load Efficiency: >80%"
echo "  • Global Store Efficiency: >80%"
echo "  • Achieved Occupancy: >50%"
echo "  • Memory Throughput: >60% of theoretical"
echo "  • Warp Execution Efficiency: >90%"
echo

echo -e "${BOLD}Optimization Checklist:${NC}"
echo "Memory Access:"
echo "  □ Coalesced global memory access (32/64/128-byte transactions)"
echo "  □ Proper memory alignment (align to transaction size)"
echo "  □ Minimize global memory transactions"
echo "  □ Use shared memory for data reuse"
echo "  □ Consider texture memory for spatial locality"
echo

echo "Kernel Configuration:"
echo "  □ Block size multiple of 32 (warp size)"
echo "  □ Typically 128-256 threads per block"
echo "  □ Launch enough blocks (>= # of SMs)"
echo "  □ Use __launch_bounds__ for register pressure"
echo "  □ Consider cooperative groups for modern patterns"
echo

echo "Compute Optimization:"
echo "  □ Use fast math intrinsics (__sinf, __cosf, etc.)"
echo "  □ Minimize divergent branches"
echo "  □ Unroll small loops with #pragma unroll"
echo "  □ Use warp-level primitives when possible"
echo "  □ Consider tensor cores for AI workloads"
echo

echo "Memory Management:"
echo "  □ Use streams for overlapping compute/transfer"
echo "  □ Use pinned memory for faster transfers"
echo "  □ Consider unified memory for simplicity"
echo "  □ Use cudaMallocAsync/cudaFreeAsync (CUDA 11.2+)"
echo "  □ Implement memory pools for frequent allocations"
echo

# Analysis recommendations
print_header "Next Steps"

echo "1. Review profiler output above for bottlenecks"
echo "2. Compare metrics against target values"
echo "3. Focus on the limiting factor (memory vs compute bound)"
echo "4. Implement optimizations based on checklist"
echo "5. Re-run analysis to verify improvements"
echo

if [ -z "$PROFILER" ]; then
    print_warning "Install CUDA profiling tools for detailed analysis:"
    echo "  • Legacy: nvprof (CUDA < 11)"
    echo "  • Modern: Nsight Compute (ncu) and Nsight Systems (nsys)"
fi

print_success "Analysis complete!"