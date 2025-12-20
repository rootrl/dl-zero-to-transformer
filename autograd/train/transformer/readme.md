(cd ../../ && python -m train.transformer.train_transformer)
Vocab Size: 13332
Model Config: Emb=64, Heads=4, Head_Size=16

=== Start MHA Sanity Check (Overfitting One Batch) ===
Step 0, Loss: 9.6602
Step 50, Loss: 6.5938
Step 100, Loss: 6.5176
Step 150, Loss: 6.1717
Step 200, Loss: 6.0586
Step 250, Loss: 5.6062
Step 300, Loss: 5.7802
Step 350, Loss: 5.5626
Step 400, Loss: 5.3528
Step 450, Loss: 5.3710
Step 500, Loss: 5.2602
Step 550, Loss: 5.1714
...
Step 4600, Loss: 3.9267
Step 4650, Loss: 3.8282
Step 4700, Loss: 3.8170
Step 4750, Loss: 3.8506
Step 4800, Loss: 3.9075
Step 4850, Loss: 3.8310
Step 4900, Loss: 3.7074
Step 4950, Loss: 3.8857

Test Complete.

=== Generating Text ===
Prompt: 'First'
Generated: Murderer:I'lltellyouwhatyouare
-----------------------
Prompt: 'Before'
Generated: youwillnot.Iwillnot,sir,
-----------------------
Prompt: 'The'
Generated: coveringsky,thekingisdead.YORK:
-----------------------
