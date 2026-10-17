"""Check that PyTorch can execute a tensor operation on an NVIDIA GPU."""

import sys

import torch


def main() -> int:
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA in PyTorch build: {torch.version.cuda}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    if not torch.cuda.is_available():
        print(
            'GPU unavailable. Check your NVIDIA driver and CUDA-enabled PyTorch installation.',
            file=sys.stderr,
        )
        return 1

    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)
    print(f'GPU: {properties.name}')
    print(f'GPU memory: {properties.total_memory / 1024**3:.1f} GiB')

    with torch.no_grad():
        x = torch.randn(1000, 1000, device=device)
        result = x @ x
        torch.cuda.synchronize(device)

    print(f'Result device: {result.device}')
    print(f'Result shape: {tuple(result.shape)}')
    print('GPU check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
