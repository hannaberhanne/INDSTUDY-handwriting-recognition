# Semester Status Report — Independent Study

**Name:** Hanna Berhane  
**Course:** CSC 450 – Independent Study  
**Advisor:** Dr. Matt Lepinski  

---

## What’s been done (Spring 2025)

This semester was mostly about getting the basics right. I built a full CNN pipeline to classify handwritten English letters using the EMNIST dataset. That meant:

- Getting the repo set up and structured (data, source, model, results)
- Training a working model on EMNIST (after realizing the dataset is rotated by default… which, rude)
- Writing prediction code for custom letter images — from load to label to output image
- Debugging preprocessing issues, path errors, file loading — all the usual chaos
- Achieving ~93% test accuracy after retraining with rotation fixes and dropout added

The model now predicts letters from individual image files pretty reliably. It saves the prediction and shows the labeled output. The system is solid.

---

## Where it’s at now

- Model is trained and saved
- Custom predictions work
- Everything’s readable and re-usable
- I finally trust my own pipeline

---

## What’s next (Summer – SURF Grant)

We’re pivoting from clean EMNIST letters to real-world data: handwritten Ghanaian records. That means:

- Preprocessing raw scans: segmenting out lines, words, or individual letters
- Adapting or retraining the model on this new data
- Dealing with messier handwriting, lower quality, maybe multiple scripts
- Outputting something usable — maybe batch predictions, CSVs, or annotated images

---

## Things I’m still figuring out

- How should we break down and label the Ghanaian records?
- Do we keep building on EMNIST or start fresh with new training data?
- What’s the best way to process a full document — by character, word, or something else?
- How do we design this to scale and actually help researchers?

---

This was a strong foundation. I got way more comfortable working with models, vision pipelines, and debugging real code. I’m excited (and a little scared) to move into the more chaotic world of real handwriting next.

**Last updated:** May 7, 2025
