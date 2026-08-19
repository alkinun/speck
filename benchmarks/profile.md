# profile

the exact-loss batch 16 profile uses one 524,288-token optimization step after two warmup steps.

- flash attention forward and backward are the largest identifiable kernel family at about 29% of gpu time
- dense matrix multiplications dominate the remaining model work
- classifier projection and softmax remain material but no longer dominate memory
- weight conversions are fused into compiled kernels and are not a major standalone cost
