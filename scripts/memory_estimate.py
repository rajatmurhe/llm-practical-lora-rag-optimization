"""Decimal GB estimates, mixed-precision AdamW, no sharding or offload."""
P=7_000_000_000
# Llama-style 32-layer 7B, hidden=4096, MLP=11008, r=16 all 7 linear projections.
r=16; layers=32; hidden=4096; mlp=11008
A=layers*r*(4*(hidden+hidden)+3*(hidden+mlp))
activation_gb=4; overhead_gb=2
for method,states in [('full',P*16/1e9),('LoRA',P*2/1e9+A*16/1e9),('QLoRA',P*0.5/1e9+P*0.05/1e9+A*16/1e9)]:
 print(method, 'persistent GB',round(states,3),'with assumed 4 GB activations + 2 GB workspace',round(states+activation_gb+overhead_gb,3))
print('adapter parameters',A)
# Example GQA inference architecture: 32 layers, 8 KV heads, head dim=128.
cache_per_token=2*32*8*128*2
print('20 concurrent users, 2048 total tokens each: KV GB',20*2048*cache_per_token/1e9)
