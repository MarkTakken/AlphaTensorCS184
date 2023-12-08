import torch


class Tokenizer():
    def __init__(self, range: tuple[int, int] = (-2, 2)) -> None:
        self.low, self.high = range
    
    def tokenize(self, tensor: torch.Tensor) -> torch.Tensor:
        dp = (tensor - self.low) * torch.Tensor([[1, (self.high-self.low) + 1, (self.high-self.low + 1)**2]])
        print(dp)
        return torch.sum(dp, axis=1).int()

    def detokenize(self, token: torch.Tensor) -> torch.Tensor:
        a = torch.remainder(token, (self.high-self.low) + 1)
        b = torch.remainder(token - a, (self.high-self.low + 1)**2) // (self.high-self.low + 1)
        c = (token - a - b) // (self.high-self.low + 1)**2
        return (torch.column_stack((a, b, c)) + self.low).int()