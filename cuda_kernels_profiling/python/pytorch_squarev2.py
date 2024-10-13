# refactored version
import torch
from abc import ABC, abstractmethod

class Square(ABC):
    @abstractmethod
    def square(self, tensor):
        pass

    def __str__(self):
        return self.__class__.__name__

class TorchSquare(Square):
    def square(self, tensor):
        return torch.square(tensor)

class Multiplication(Square):
    def square(self, tensor):
        return tensor * tensor

class Power(Square):
    def square(self, tensor):
        return tensor ** 2

class SquarePerformanceTester:
    def __init__(self, strategies):
        self.strategies = strategies

    def time_strategy(self, strategy, input_tensor):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(5):
            strategy.square(input_tensor)
        
        start.record()
        strategy.square(input_tensor)
        end.record()
        
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def profile_strategy(self, strategy, input_tensor):
        print(f"=============")
        print(f"Profiling {strategy}")
        print(f"=============")
        with torch.profiler.profile() as prof:
            strategy.square(input_tensor)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def run_tests(self, input_tensor):
        for strategy in self.strategies:
            elapsed_time = self.time_strategy(strategy, input_tensor)
            print(f"{strategy} elapsed time: {elapsed_time:.4f} ms")
            self.profile_strategy(strategy, input_tensor)

if __name__ == "__main__":
    # Setup
    torch.cuda.init()
    input_tensor = torch.randn(10000, 10000).cuda()

    # Create strategies
    strategies = [
        TorchSquare(),
        Multiplication(),
        Power()
    ]

    # Run tests
    tester = SquarePerformanceTester(strategies)
    tester.run_tests(input_tensor)