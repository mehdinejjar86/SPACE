#!/usr/bin/env python3
# smoke_test.py
# Test SPACE model (temporal interpolation at original resolution)

import torch
import sys


def test_space():
    """Test SPACE model."""
    print("=" * 60)
    print("Testing SPACE Model")
    print("=" * 60)

    from model import SPACE, build_space

    # Use tiny config for fast testing
    print("\n1. Building SPACE (tiny config)...")
    model = build_space('tiny')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Print parameter count
    params = model.get_param_count()
    print(f"   Total parameters: {params['total']:,}")
    for name, count in params.items():
        if name != 'total':
            print(f"     {name}: {count:,}")

    # Test inputs
    B, N = 2, 4
    H, W = 64, 64

    frames = torch.randn(B, N, 3, H, W, device=device)
    times = torch.tensor([[0.0, 0.33, 0.66, 1.0]] * B, device=device)

    print(f"\n2. Input: {B} batches, {N} frames, {H}x{W} resolution")

    # Test encoding
    print("\n3. Testing encode()...")
    query_time = torch.tensor([0.5] * B, device=device)
    encoding = model.encode(frames, times, query_time)
    print(f"   scene_code: {encoding['scene_code'].shape}")
    print(f"   feature_grids: {len(encoding['feature_grids'])} scales")
    for i, g in enumerate(encoding['feature_grids']):
        print(f"     scale {i}: {g.shape}")
    print(f"   biases: {len(encoding['biases'])} layers")
    print("   PASSED!")

    # Test coordinate query
    print("\n4. Testing coordinate query...")
    Q = 512
    coords = torch.rand(B, Q, 3, device=device) * 2 - 1
    coords[:, :, 2] = 0.5
    rgb = model(frames, times, coords)
    print(f"   coords: {coords.shape} -> rgb: {rgb.shape}")
    assert rgb.shape == (B, Q, 3)
    print("   PASSED!")

    # Test HQ rendering (output matches input resolution)
    print("\n5. Testing render_frame(mode='hq')...")
    out = model.render_frame(frames, times, target_time=0.5, mode='hq')
    print(f"   Output: {out.shape} (matches input {H}x{W})")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test fast rendering (output matches input resolution)
    print("\n6. Testing render_frame(mode='fast')...")
    out = model.render_frame(frames, times, target_time=0.5, mode='fast')
    print(f"   Output: {out.shape} (matches input {H}x{W})")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test auto mode
    print("\n7. Testing render_frame(mode='auto')...")
    out = model.render_frame(frames, times, target_time=0.5, mode='auto')
    print(f"   Output: {out.shape}")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test chunked rendering
    print("\n8. Testing render_frame_chunked()...")
    out = model.render_frame_chunked(frames, times, target_time=0.5, chunk_size=256)
    print(f"   Output: {out.shape} (matches input {H}x{W})")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test gradient flow
    print("\n9. Testing gradient flow...")
    model.zero_grad()
    out = model.render_frame(frames, times, target_time=0.5, mode='hq')
    loss = out.mean()
    loss.backward()

    has_grad = {name: any(p.grad is not None for p in module.parameters())
                for name, module in [('encoder', model.encoder),
                                     ('aggregator', model.aggregator),
                                     ('decoder', model.decoder)]}
    for name, has in has_grad.items():
        status = 'OK' if has else 'NO GRAD!'
        print(f"   {name}: {status}")
        assert has
    print("   PASSED!")

    print("\n" + "=" * 60)
    print("ALL SPACE TESTS PASSED!")
    print("=" * 60)


def test_loss_functions():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    from model.loss_x import SPACELoss, CharbonnierLoss, SSIMLoss, FrequencyLoss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = torch.rand(2, 3, 64, 64, device=device)
    target = torch.rand(2, 3, 64, 64, device=device)

    # Test individual losses
    print("\n1. Testing CharbonnierLoss...")
    loss = CharbonnierLoss()(pred, target)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. Testing SSIMLoss...")
    loss = SSIMLoss()(pred, target)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. Testing FrequencyLoss...")
    loss = FrequencyLoss()(pred, target)
    print(f"   Loss: {loss.item():.4f}")

    print("\n4. Testing SPACELoss (combined)...")
    criterion = SPACELoss(use_perceptual=False)  # Skip VGG for speed
    losses = criterion(pred, target)
    print(f"   Total: {losses['total'].item():.4f}")
    print(f"   Charbonnier: {losses['charbonnier'].item():.4f}")
    print(f"   SSIM: {losses['ssim'].item():.4f}")
    print(f"   Frequency: {losses['frequency'].item():.4f}")

    print("\n5. Testing with coordinate format [B, Q, 3]...")
    pred_coords = torch.rand(2, 1024, 3, device=device)
    target_coords = torch.rand(2, 1024, 3, device=device)
    losses = criterion(pred_coords, target_coords)
    print(f"   Total: {losses['total'].item():.4f}")

    print("\nAll loss tests passed!")


def test_model_configs():
    """Test different model sizes."""
    print("\n" + "=" * 60)
    print("Testing Model Configurations")
    print("=" * 60)

    from model import build_space

    for config in ['tiny', 'base']:
        print(f"\n{config.upper()} config:")
        model = build_space(config)
        params = model.get_param_count()
        print(f"  Total: {params['total']:,} parameters")


def test_n4_768():
    """Test N=4 with 768x768 resolution."""
    print("\n" + "=" * 60)
    print("Testing N=4 with 768x768 Resolution")
    print("=" * 60)

    from model import build_space
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU, this will be slow!")

    # Build base model for this test
    print("\n1. Building SPACE (base config)...")
    model = build_space('base')
    model = model.to(device)
    model.eval()

    params = model.get_param_count()
    print(f"   Total parameters: {params['total']:,}")

    # Test with N=4 and 768x768
    B, N = 1, 4
    H, W = 768, 768

    print(f"\n2. Creating input: B={B}, N={N}, {H}x{W}")
    frames = torch.randn(B, N, 3, H, W, device=device)
    times = torch.tensor([[0.0, 0.33, 0.67, 1.0]] * B, device=device)

    # Check memory before
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU memory before: {mem_before:.2f} GB")

    # Test encoding
    print("\n3. Testing encode()...")
    with torch.no_grad():
        start = time.time()
        query_time = torch.tensor([0.5] * B, device=device)
        encoding = model.encode(frames, times, query_time)
        encode_time = time.time() - start
    print(f"   Encode time: {encode_time:.3f}s")
    print(f"   scene_code: {encoding['scene_code'].shape}")
    print(f"   feature_grids: {len(encoding['feature_grids'])} scales")
    for i, g in enumerate(encoding['feature_grids']):
        print(f"     scale {i}: {g.shape}")

    # Test render_frame (fast mode)
    print("\n4. Testing render_frame(mode='fast')...")
    with torch.no_grad():
        start = time.time()
        out = model.render_frame(frames, times, target_time=0.5, mode='fast')
        render_time = time.time() - start
    print(f"   Render time: {render_time:.3f}s")
    print(f"   Output: {out.shape}")
    assert out.shape == (B, 3, H, W), f"Expected {(B, 3, H, W)}, got {out.shape}"
    print("   PASSED!")

    # Check memory after
    if device == 'cuda':
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n5. GPU memory peak: {mem_peak:.2f} GB")

    # Test with different target times (simulating cascaded prediction)
    print("\n6. Testing cascaded-style prediction...")
    target_times = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
    with torch.no_grad():
        start = time.time()
        for t in target_times:
            out = model.render_frame(frames, times, target_time=t, mode='fast')
        total_time = time.time() - start
    print(f"   7 frames rendered in {total_time:.3f}s ({total_time/7:.3f}s per frame)")

    print("\n" + "=" * 60)
    print("N=4 768x768 TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n4-768', action='store_true', help='Run N=4 768x768 test only')
    parser.add_argument('--all', action='store_true', help='Run all tests including N=4 768x768')
    args = parser.parse_args()

    try:
        if args.n4_768:
            # Run only the N=4 768x768 test
            test_n4_768()
        elif args.all:
            # Run all tests
            test_space()
            test_loss_functions()
            test_model_configs()
            test_n4_768()
        else:
            # Default: run basic tests (fast)
            test_space()
            test_loss_functions()
            test_model_configs()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
