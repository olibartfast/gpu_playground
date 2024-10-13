#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

class Square {
public:
    virtual torch::Tensor square(const torch::Tensor& tensor) = 0;
    virtual ~Square() = default;
    virtual std::string name() const = 0;
};

class TorchSquare : public Square {
public:
    torch::Tensor square(const torch::Tensor& tensor) override {
        return torch::square(tensor);
    }
    std::string name() const override { return "TorchSquare"; }
};

class Multiplication : public Square {
public:
    torch::Tensor square(const torch::Tensor& tensor) override {
        return tensor * tensor;
    }
    std::string name() const override { return "Multiplication"; }
};

class Power : public Square {
public:
    torch::Tensor square(const torch::Tensor& tensor) override {
        return torch::pow(tensor, 2);
    }
    std::string name() const override { return "Power"; }
};

class SquarePerformanceTester {
private:
    std::vector<std::unique_ptr<Square>> strategies;

    double time_strategy(Square* strategy, const torch::Tensor& input) {
        // Warmup
        for (int i = 0; i < 5; ++i) {
            strategy->square(input);
        }

        auto start = torch::cuda::Event(true);
        auto end = torch::cuda::Event(true);

        start.record();
        auto result = strategy->square(input);
        end.record();

        torch::cuda::synchronize();

        return start.elapsed_time(end);
    }

public:
    SquarePerformanceTester(std::vector<std::unique_ptr<Square>> strats) 
        : strategies(std::move(strats)) {}

    void run_tests(const torch::Tensor& input) {
        for (const auto& strategy : strategies) {
            double elapsed_time = time_strategy(strategy.get(), input);
            std::cout << strategy->name() << " elapsed time: " 
                      << elapsed_time << " ms" << std::endl;

            // Profiling
            std::cout << "=============" << std::endl;
            std::cout << "Profiling " << strategy->name() << std::endl;
            std::cout << "=============" << std::endl;
            
            torch::autograd::profiler::RecordProfile guard;
            strategy->square(input);
        }
    }
};

int main() {
    torch::manual_seed(1);

    // Check if CUDA is available
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Setup
    torch::Tensor input = torch::randn({10000, 10000}, device);

    // Create strategies
    std::vector<std::unique_ptr<Square>> strategies;
    strategies.push_back(std::make_unique<TorchSquare>());
    strategies.push_back(std::make_unique<Multiplication>());
    strategies.push_back(std::make_unique<Power>());

    // Run tests
    SquarePerformanceTester tester(std::move(strategies));
    tester.run_tests(input);

    return 0;
}