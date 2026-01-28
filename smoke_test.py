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

    # Test auto mode (should use NAFNet by default)
    print("\n7. Testing render_frame(mode='auto')...")
    out = model.render_frame(frames, times, target_time=0.5, mode='auto')
    print(f"   Output: {out.shape}")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test hybrid rendering explicitly (now the default)
    print("\n7b. Testing render_frame(mode='hybrid')...")
    out = model.render_frame(frames, times, target_time=0.5, mode='hybrid')
    print(f"   Output: {out.shape}")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test chunked rendering
    print("\n8. Testing render_frame_chunked()...")
    out = model.render_frame_chunked(frames, times, target_time=0.5, chunk_size=256)
    print(f"   Output: {out.shape} (matches input {H}x{W})")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test gradient flow (HQ mode)
    print("\n9. Testing gradient flow (HQ mode)...")
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

    # Test gradient flow (Hybrid mode)
    print("\n10. Testing gradient flow (Hybrid mode)...")
    model.zero_grad()
    out = model.render_frame(frames, times, target_time=0.5, mode='hybrid')
    loss = out.mean()
    loss.backward()

    has_grad = {name: any(p.grad is not None for p in module.parameters())
                for name, module in [('encoder', model.encoder),
                                     ('aggregator', model.aggregator),
                                     ('hybrid_decoder', model.hybrid_decoder)]}
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


def test_nafnet_modulation():
    """Test NAFNet decoder with scene_code modulation."""
    print("\n" + "=" * 60)
    print("Testing NAFNet Scene-Code Modulation")
    print("=" * 60)

    from model.nafnet_decoder import NAFNetDecoder
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test with modulation
    print("\n1. Testing NAFNetDecoder with FiLM modulation...")
    decoder = NAFNetDecoder(
        feature_dims=[32, 64, 128, 256],
        hidden_dim=32,
        num_blocks=[1, 1, 2, 2],
        mod_dim=128,  # Enable modulation
    ).to(device)

    # Create fake multi-scale features
    B = 2
    feature_grids = [
        torch.randn(B, 32, 32, 32, device=device),   # Scale 0
        torch.randn(B, 64, 16, 16, device=device),   # Scale 1
        torch.randn(B, 128, 8, 8, device=device),    # Scale 2
        torch.randn(B, 256, 4, 4, device=device),    # Scale 3
    ]
    scene_code = torch.randn(B, 128, device=device)

    # Forward with modulation
    out = decoder(feature_grids, scene_code)
    print(f"   Input features: {[f.shape for f in feature_grids]}")
    print(f"   Scene code: {scene_code.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape[0] == B and out.shape[1] == 3
    print("   PASSED!")

    # Test that different scene_codes produce different outputs
    print("\n2. Testing modulation effect...")
    scene_code_1 = torch.randn(B, 128, device=device)
    scene_code_2 = torch.randn(B, 128, device=device)

    out_1 = decoder(feature_grids, scene_code_1)
    out_2 = decoder(feature_grids, scene_code_2)

    diff = (out_1 - out_2).abs().mean().item()
    print(f"   Output difference with different scene_codes: {diff:.6f}")
    assert diff > 0.001, "Modulation should produce different outputs"
    print("   PASSED!")

    # Test without modulation (scene_code=None)
    print("\n3. Testing NAFNetDecoder without modulation...")
    decoder_no_mod = NAFNetDecoder(
        feature_dims=[32, 64, 128, 256],
        hidden_dim=32,
        num_blocks=[1, 1, 2, 2],
        mod_dim=None,  # No modulation
    ).to(device)

    out_no_mod = decoder_no_mod(feature_grids, None)
    print(f"   Output (no modulation): {out_no_mod.shape}")
    assert out_no_mod.shape[0] == B and out_no_mod.shape[1] == 3
    print("   PASSED!")

    # Test gradient flow through modulation
    print("\n4. Testing gradient flow through modulation...")
    decoder.zero_grad()
    out = decoder(feature_grids, scene_code)
    loss = out.mean()
    loss.backward()

    # Check that FiLM layer received gradients
    has_film_grad = decoder.film.weight.grad is not None
    print(f"   FiLM layer gradient: {'OK' if has_film_grad else 'NO GRAD!'}")
    assert has_film_grad
    print("   PASSED!")

    print("\nAll NAFNet modulation tests passed!")


def test_hybrid_decoder():
    """Test HybridNAFNetSIREN decoder specifically."""
    print("\n" + "=" * 60)
    print("Testing Hybrid NAFNet+SIREN Decoder")
    print("=" * 60)

    from model.nafnet_decoder import HybridNAFNetSIREN
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n1. Testing HybridNAFNetSIREN...")
    decoder = HybridNAFNetSIREN(
        feature_dims=[32, 64, 128, 256],
        hidden_dim=32,
        num_blocks=[1, 1, 2, 2],
        latent_dim=128,
        siren_hidden_dim=32,
        siren_num_layers=2,
    ).to(device)

    B = 2
    feature_grids = [
        torch.randn(B, 32, 32, 32, device=device),
        torch.randn(B, 64, 16, 16, device=device),
        torch.randn(B, 128, 8, 8, device=device),
        torch.randn(B, 256, 4, 4, device=device),
    ]
    scene_code = torch.randn(B, 128, device=device)

    # Full-frame forward (no coords provided)
    out = decoder(feature_grids, scene_code, coords=None)
    print(f"   Full-frame output: {out.shape}")
    assert out.shape[0] == B and out.shape[1] == 3
    print("   PASSED!")

    # With explicit coordinates
    print("\n2. Testing with explicit coordinates...")
    H, W = 64, 64
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([
        xx.flatten().unsqueeze(0).expand(B, -1),
        yy.flatten().unsqueeze(0).expand(B, -1),
        torch.full((B, H * W), 0.5, device=device),
    ], dim=-1)

    out = decoder(feature_grids, scene_code, coords=coords)
    print(f"   With coords output: {out.shape}")
    assert out.shape == (B, 3, H, W)
    print("   PASSED!")

    # Test coordinate-based forward
    print("\n3. Testing forward_coords (for training)...")
    Q = 512
    coords_sparse = torch.rand(B, Q, 3, device=device) * 2 - 1
    out_sparse = decoder.forward_coords(feature_grids, scene_code, coords_sparse)
    print(f"   Sparse coords output: {out_sparse.shape}")
    assert out_sparse.shape == (B, Q, 3)
    print("   PASSED!")

    # Gradient flow
    print("\n4. Testing gradient flow...")
    decoder.zero_grad()
    out = decoder(feature_grids, scene_code)
    loss = out.mean()
    loss.backward()

    # Check both NAFNet and SIREN parts receive gradients
    nafnet_grad = any(p.grad is not None for p in decoder.nafnet.parameters())
    siren_grad = any(p.grad is not None for p in decoder.siren_guidance.parameters())
    print(f"   NAFNet gradients: {'OK' if nafnet_grad else 'NO GRAD!'}")
    print(f"   SIREN gradients: {'OK' if siren_grad else 'NO GRAD!'}")
    assert nafnet_grad and siren_grad
    print("   PASSED!")

    # Check blend_alpha is learning
    print(f"\n5. Blend alpha: {decoder.blend_alpha.item():.4f}")
    has_alpha_grad = decoder.blend_alpha.grad is not None
    print(f"   Blend alpha gradient: {'OK' if has_alpha_grad else 'NO GRAD!'}")
    print("   PASSED!")

    print("\nAll Hybrid decoder tests passed!")


def test_hybrid_loss():
    """Test HybridLoss for full-image training."""
    print("\n" + "=" * 60)
    print("Testing HybridLoss")
    print("=" * 60)

    from model.loss_advanced import HybridLoss
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize loss (skip LPIPS for speed)
    print("\n1. Initializing HybridLoss...")
    criterion = HybridLoss(
        use_charbonnier=True,
        use_msssim=True,
        use_lpips=False,  # Skip for speed
        use_gradient=True,
        use_frequency=True,
        use_anchor_distillation=False,  # Test without anchor for now
        use_coord_loss=True,
        use_uncertainty_weighting=True,
        use_progressive=True,
    ).to(device)
    print("   PASSED!")

    # Test inputs
    B = 2
    H, W = 64, 64
    Q = 512

    pred_full = torch.rand(B, 3, H, W, device=device)
    target_full = torch.rand(B, 3, H, W, device=device)
    pred_coords = torch.rand(B, Q, 3, device=device)
    target_coords = torch.rand(B, Q, 3, device=device)

    # Test full-frame only
    print("\n2. Testing full-frame loss...")
    losses = criterion(pred_full, target_full)
    print(f"   Total: {losses['total'].item():.4f}")
    print(f"   Losses: {[k for k in losses.keys() if not k.startswith('_')]}")
    assert 'total' in losses
    assert 'charbonnier' in losses
    print("   PASSED!")

    # Test with coordinates
    print("\n3. Testing with coordinate loss...")
    losses = criterion(pred_full, target_full, pred_coords, target_coords)
    print(f"   Total: {losses['total'].item():.4f}")
    if 'coord_loss' in losses:
        print(f"   Coord loss: {losses['coord_loss'].item():.4f}")
    print("   PASSED!")

    # Test progressive curriculum
    print("\n4. Testing progressive curriculum...")
    for epoch in [0, 5, 15, 30, 50]:
        active = criterion.update_for_epoch(epoch)
        print(f"   Epoch {epoch}: {active}")
    print("   PASSED!")

    # Test uncertainty weighting
    print("\n5. Testing uncertainty weighting...")
    stats = criterion.get_uncertainty_stats()
    if stats:
        print(f"   Weights: {stats['weights']}")
        print(f"   Sigmas: {stats['sigmas']}")
    print("   PASSED!")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    criterion.zero_grad()
    losses = criterion(pred_full, target_full, pred_coords, target_coords)
    losses['total'].backward()

    # Check uncertainty params received gradients
    if hasattr(criterion, 'uncertainty'):
        has_unc_grad = any(p.grad is not None for p in criterion.uncertainty.parameters())
        print(f"   Uncertainty gradients: {'OK' if has_unc_grad else 'NO GRAD'}")

    print("   PASSED!")

    print("\nAll HybridLoss tests passed!")


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
        # Check that hybrid decoder is enabled by default
        assert model.use_hybrid_decoder, f"{config} should use hybrid decoder by default"
        print(f"  Hybrid decoder: enabled")


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


def test_4k_inference():
    """Test inference at 4K resolution (4096x2160)."""
    print("\n" + "=" * 60)
    print("Testing 4K Inference (4096x2160)")
    print("=" * 60)

    from model import build_space
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU, this will be VERY slow!")
        print("         Consider running on GPU cluster.")

    # Build base model
    print("\n1. Building SPACE (base config)...")
    model = build_space('base')
    model = model.to(device)
    model.eval()

    params = model.get_param_count()
    print(f"   Total parameters: {params['total']:,}")

    # Test with N=2 (typical for 8x interpolation start) and 4K resolution
    B, N = 1, 2
    H, W = 2160, 4096  # 4K UHD

    print(f"\n2. Creating input: B={B}, N={N}, {W}x{H} (4K)")

    # Check memory before allocation
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU memory before: {mem_before:.2f} GB")

    frames = torch.randn(B, N, 3, H, W, device=device)
    times = torch.tensor([[0.0, 1.0]] * B, device=device)

    if device == 'cuda':
        mem_after_input = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU memory after input: {mem_after_input:.2f} GB")

    # Test encoding
    print("\n3. Testing encode()...")
    with torch.no_grad():
        start = time.time()
        query_time = torch.tensor([0.5] * B, device=device)
        encoding = model.encode(frames, times, query_time)
        if device == 'cuda':
            torch.cuda.synchronize()
        encode_time = time.time() - start
    print(f"   Encode time: {encode_time:.3f}s")
    print(f"   scene_code: {encoding['scene_code'].shape}")
    print(f"   feature_grids: {len(encoding['feature_grids'])} scales")
    for i, g in enumerate(encoding['feature_grids']):
        print(f"     scale {i}: {g.shape}")

    # Test render_frame (fast mode) - this is the key test
    print("\n4. Testing render_frame(mode='fast') at 4K...")
    with torch.no_grad():
        start = time.time()
        out = model.render_frame(frames, times, target_time=0.5, mode='fast')
        if device == 'cuda':
            torch.cuda.synchronize()
        render_time = time.time() - start
    print(f"   Render time: {render_time:.3f}s")
    print(f"   Output: {out.shape}")
    assert out.shape == (B, 3, H, W), f"Expected {(B, 3, H, W)}, got {out.shape}"
    print("   PASSED!")

    # Check memory peak
    if device == 'cuda':
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n5. GPU memory peak: {mem_peak:.2f} GB")

    # Test chunked rendering (memory-efficient alternative)
    print("\n6. Testing render_frame_chunked() at 4K...")
    with torch.no_grad():
        start = time.time()
        out_chunked = model.render_frame_chunked(
            frames, times, target_time=0.5, chunk_size=1024
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        chunked_time = time.time() - start
    print(f"   Chunked render time: {chunked_time:.3f}s")
    print(f"   Output: {out_chunked.shape}")
    assert out_chunked.shape == (B, 3, H, W)
    print("   PASSED!")

    # Verify outputs are similar (chunked vs fast)
    diff = (out - out_chunked).abs().mean()
    print(f"   Difference (fast vs chunked): {diff.item():.6f}")

    print("\n" + "=" * 60)
    print("4K INFERENCE TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n4-768', action='store_true', help='Run N=4 768x768 test only')
    parser.add_argument('--4k', dest='test_4k', action='store_true', help='Run 4K inference test only')
    parser.add_argument('--all', action='store_true', help='Run all tests including high-res')
    args = parser.parse_args()

    try:
        if args.n4_768:
            # Run only the N=4 768x768 test
            test_n4_768()
        elif args.test_4k:
            # Run only the 4K inference test
            test_4k_inference()
        elif args.all:
            # Run all tests
            test_space()
            test_hybrid_decoder()
            test_nafnet_modulation()
            test_loss_functions()
            test_hybrid_loss()
            test_model_configs()
            test_n4_768()
            test_4k_inference()
        else:
            # Default: run basic tests (fast)
            test_space()
            test_hybrid_decoder()
            test_nafnet_modulation()
            test_loss_functions()
            test_hybrid_loss()
            test_model_configs()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
