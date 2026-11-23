# Beta support

| Model | Dedicated B1 | Fixed batch | Online default | Explicit alternatives |
| --- | --- | --- | --- | --- |
| π0.5 H50 | Dedicated Prefix FP8 on the measured G4, BF16 on the other three measured GPUs | Exact powers of two through the selected cap | G4 Prefix FP8 cap 16, local FIFO 0 ms, binary server FIFO 2 ms. RTX 4090/3090 cap 4, 0 ms. A100 80 GB cap 4, 2 ms. | BF16 reference, portable BF16, supported named precision recipes. OpenPI H10 uses its own BF16 route. |
| OpenVLA-OFT | BF16 | Exact widths through the selected cap | BF16 cap 4, wait 2 ms on G4/A100/3090 and 0 ms on 4090. | Mixed FP8 on tested G4/4090 with the matching FA2 operator. The full language projection scope is selected by --recipe mixed-fp8. |
| LingBot-VLA2 | Mixed cache32 | TrueBatch, B4 by default in fixed mode | Mixed cache32, B1, 0 ms, including when eight clients are connected | --recipe split-stage gives an experimental multi-width engine. BF16 and historical precision recipes remain available. |

G4 denotes the 96 GB RTX PRO 6000 Blackwell Server Edition. A GPU with the same SM can have a different kernel or memory contract. The worker reports total/free memory, selected profile, resolved recipe, native probe and fallback reasons. OFT drops expensive packing as free memory falls. A new card uses a conservative π0.5 route. Explicit unsupported FP8 produces an error. Older 24 GB consumer cards can run the mixed LingBot worker, but upstream FP32 memory requirements and simulation/renderer memory are separate.

### Full-checkpoint results

All three default routes ran from the five beta 1 wheels on each card. All 96 frozen B1 action checks passed their model-specific criteria. Each local C8 entry below counts chunks returned per second in a warmed 7-second smoke window, in model order π0.5 / OFT / LingBot.

| GPU | Local C8, beta 1 wheels |
| --- | --- |
| RTX PRO 6000 | 101.71 / 36.57 / 10.57 |
| A100 80 GB | 43.43 / 21.14 / 6.86 |
| RTX 4090 | 32.57 / 14.29 / 8.00 |
| RTX 3090 | 17.86 / 6.29 / 6.29 |

Subsequent three-window measurements with those wheels put LingBot cache32 at 8.23 and 6.22 chunks/s on RTX 4090 and 3090. The explicit split-stage route reached 25.47 on G4 and 13.87 on A100 in the report. On G4 it achieved 37/40 on held-out RoboTwin initial states, gaining two successes and losing two against cache32. Beta 2 uses the same model and native kernel implementations. The packaged smoke results refer to beta 1. [Online profiles and measurement fingerprints](../src/vvla/data/online-profiles.json)
