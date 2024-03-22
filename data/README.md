## Training data

The model is trained and tested on [mnist dataset](http://yann.lecun.com/exdb/mnist/). 

## Testing results

Networks we trained on a set of same 10000 images.
They were tested on set of 27730 unseen images.

| Architecture | Training time | Test  |
|--------------|---------------|-------|
| 2L 64n       | 20h 5m 31s    | 53.0% |
| 3L 16N       | 5h 46m 27s    | 24.7% |
| 3L 24N       | 8h 30m 19s    | 34.6% |
| 3L 32N       | 10h 42m 14s   | 54.5% |
| 3L 40N       | 13h 49m 14s   | 50.9% |
| 3L 48N       | 16h 48m 6s    | 57.9% |
| 3L 56N       | 19h 54m 12s   | 55.2% |
| 3L 64N       | 23h 25m 54s   | 58.4% |