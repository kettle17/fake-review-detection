The Experiment_Review dataset has 200 entries from the Salminen dataset and 200 entries from the FRDDS dataset (400 total)
There is a 200:200 split between real and AI-generated reviews, with 40 reviews each category.
These are for human testing.
The point of the experiment is to:

1. Evaluate if human accuracy has improved since the AI boom/Salminen paper (end of 2020 to mid 2024)
2. Evaluate if humans are better at detecting GPT-2 or GPT-4 reviews (Salminen/FRDDS)
3. Evaluate if a certain age group is better at detecting AI reviews.

The full unaltered dataset is in Experiment_Reviews.csv
However subjects were given the ForHuman_Experiment_Reviews.csv instead for testing purposes:
	Subjects were not told if reviews were AI or human.
	The only information taken from both datasets were the review text and rating sentiment (1-5).
	They were not told which dataset each review belongs to.
	The only information taken about each subject was their age bracket (under 40 or over 40).

Subjects 1, 3, and 6 belong to the under 40 bracket
Subjects 2, 4, and 5 belong to the over 40 bracket