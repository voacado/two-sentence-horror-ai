<a href="https://www.reddit.com/r/TwoSentenceHorror/">
    <img src="https://styles.redditmedia.com/t5_30tmh/styles/bannerBackgroundImage_dvex47b9l3g61.jpg?format=pjpg&s=2e469a7172684dcbe60b8c09493af6e89c5638d1" alt="Two Sentence Horror banner" title="Two Sentence Horror" align="right" height="60" />
</a>

# Two Sentence Horror LM
> A language model that generate two-sentence horror stories 👻

## Quick Links:
<div id="badges">
  <a href="https://www.kaggle.com/datasets/voanthony/two-sentence-horror-jan-2015-apr-2023">
    <img src="https://img.shields.io/badge/Dataset-4FCB93?style=for-the-badge" alt="Dataset Badge"/>
  </a>
  <a href="https://www.kaggle.com/code/voanthony/two-sentence-horror-bart-model">
    <img src="https://img.shields.io/badge/BART%20Run%20%28Kaggle%29-267DAE?style=for-the-badge" alt="Code Run, BART (Kaggle) Badge"/>
  </a>
  <a href="https://huggingface.co/voacado/bart-two-sentence-horror">
    <img src="https://img.shields.io/badge/BART%20Model%20%28Hugging%20Face%29-FFD228?style=for-the-badge" alt="Model, BART (Hugging Face) Badge"/>
  </a>
  <a href="https://huggingface.co/spaces/voacado/two-sentence-horror-bart">
    <img src="https://img.shields.io/badge/Interactive%20Space%20%28Hugging%20Face%29-FF3270?style=for-the-badge" alt="Online Space, BART (Hugging Face) Badge"/>
  </a>
</div>

## What is "Two-Sentence Horror"?
Two-Sentence Horror is a creative form of storytelling where users attempt to scare the reader in a short medium. In this format, the narrative is developed over just two sentences, challenging writers to evoke fear, suspense, and tension in a limited amount of words. The first sentence typically sets the stage or introduces a seemingly normal situation, while the second sentence delivers a horrifying and usually unexpected twist, leaving the reader uneasy and in shock. This minimalist form of storytelling emphasizes the potency of short-form narratives, greatly leaning into the ability for people to imagine. This is even more important nowadays as most forms of media have become short-form, ranging from social media posts to YouTube videos rather than reading books.

Two-Sentence Horror is primarily found on a subreddit named r/TwoSentenceHorror. They have roughly 1.3 million subscribers with an average of 200 to 300 post a day. As of December 2023, they are the 447th most popular subreddit. Given the Reddit user base of 430 million monthly active users and 52 million active daily users (as of 2019), r/TwoSentenceHorror is, all things considered, a very active subreddit.

### Motivations
Two-Sentence Horror is a rather unexplored field for text generation. As of December 2023, there is one other research paper (titled “HorrifAI: Using AI to Generate Two-Sentence Horror”) that attempts to create a similar language model. During exploration, we found no publicly available dataset for Two-Sentence Horror that we deemed usable for our models. Given the sheer scale of the subreddit r/TwoSentenceHorror, we figured this would be a good opportunity to scrape real data from user input.

Additionally, language models have historically been relatively poor at understanding human emotion. We believed it would be interesting to see if a model could understand how to invoke fear in a reader from these horror stories.

## The Objective

We aim to create a language model capable of understanding and emulating horror as an emotion, creating two-sentence horror stories that make sense and invoke a sense of fear in readers. The goal is for users to be able to input a first sentence to the horror story, and the language model generates an appropriate second sentence to complete it.

## Dataset
<i>(Note, the following text below is extracted from the final report. Please reference `CS4120 (NLP) - Final Project Report.pdf` for additional information.)</i>

Our dataset, which we named `Two Sentence Horror (Jan 2015 - Apr 2023)`, is something that we created from scratch. The original data contains about 107K user posts, but after data cleansing, we had about 95K posts available to train and fine-tune with.

The dataset was collected from the subreddit `r/TwoSentenceHorror` using the `PullPush.io API` supplied by Reddit over the course of roughly 1100 GET requests. With a frequency of 200 to 300 new stories a day, extracting all data would become overwhelming, especially when most of the stories submitted are very low-quality. Thus, our criteria for a post was that it had at least 21 or more upvotes and was posted after January 1st, 2015. From observation, stories eitherreceived none or over 21 upvotes. We decided that this metric would be a relatively okay cut-off for determining the quality of a story.

When extracting, we looked for five specific elements of a Reddit post. Primarily, the post title and selftext contained the story itself (with the title being the first sentence and the selftext being the second). We also extracted metrics evaluating a post’s performance - its score (number of upvotes), num_comments (number of comments), and gilded_count (number of awards received).

The data was extracted using Python’s requests library (file reference: `dataset/code/scrape_data.ipynb`). It was stored into a CSV file and is publicly available on Kaggle listed as the dataset “Two Sentence Horror (Jan 2015 - Apr 2023)”. The original data is named `org_reddit_scrape_20_Jan2015_timestamp.csv`.

### Exploration of Dataset

While we extracted 107K user posts, only about 95K of those posts were usable for our model. While exploring the dataset, we noticed common trends in certain user posts that we could deduce were not either not a story or not fit for our model. For one, many Reddit tags are surrounded by `[` and `]` tokens, such as `[removed]` or `[MAY2020]`. Posts that were deleted were removed from the dataset completely, whereas posts that contained tags remained with the tags removed (refer to `dataset/code/process_data.ipynb`).

While removing 12K posts may seem like a lot after this first step of data cleansing, it is important to note that between May and July 2023, a change on Reddit’s API pricing and availability caused a protest of many Reddit users. Besides causing a site-wide blackout, many users deleted their accounts in protest, resulting in a greater influx of `[removed]` and `[deleted]` posts.

After removing all posts that seemed to be invalid as well as tags in the texts of posts, we originally proceeded with attempting to train our models but frequently ran into either memory problems or only predicting padding tokens. Upon further observation, our dataset had a very imbalanced distribution of token lengths.

While the longest selftext was 461 tokens long (all sequences were padded to the longest length for both title and selftext), 99.76% of the data was below 50 tokens, 98.83% was below 40 tokens, and 92.18% was below 30 tokens. The result was a significant waste of memory with sequences being over-padded for seemingly no benefit. We made the design decision to remove all posts from our dataset above 40 tokens as we considered that a good balance between retaining the context of r/TwoSentenceHorror while also saving significantly on memory usage while training.

## Methods and Results

For this language model, we took four different approaches at creating this language model, starting from the most basic of model architectures and working our way up to see what created a more potent model.

#### Method 1: Keras Sequential
Initially, we began with a `Keras Sequential` model that utilized n-grams and word vector embeddings to probabilistically select the next token (by essentially order of frequency). The output here were stories that seemed horrifying in nature, but conveyed no actual meaning as probabilistic selection was not capable of capturing the meaning of tokens properly:

- Ex 1: **There was a ghost**. Wait, - was rather me.
- Ex 2: **I was horrified when I got my tests back.** I’d put a man to said a lights toward me, their murders is them.
- Ex 3: **My parents told me not to go upstairs.** I've been terrible, it was brought a odd way over my lie, , she know they can stop, I wonder how deep my ankle.
- Ex 4: **I got out of bed this morning.** For two better for to scream as something now before the dark, they came from next.

#### Method 2: Seq2Seq
The next step in model exploration was adding in a way to properly capture token semantic meaning in relation to each other. Thus, we employed a `Seq2Seq` model architecture that created vectors between the encoder (first sentence) and decoder (second sentence) so that we properly had a way to establish that "x" word in input usually results in "y" word in output. The outputs unfortunately did not make much sense:

- Ex 1: **I got out of bed this morning.** by the limb struggling that malnourished i dropped you understand that the scopes didn't stop but 2 as instead
- Ex 2: **I was horrified when I get my test results back.** i begged the answer blessedly coming to eat my flesh i an' “i have realise that where not not never who ”
- Ex 3: **My parents told me not to go upstairs.** and something was traffic champagne tellers and never confused crawling off the chair on hair of screams
- Ex 4: **There was a ghost.** the needs hershey’s slowly blood at miles from allowed full of and screaming and won’t keep picking me and approaching the chance smiling away her window

The main flaw here we believe was that, in horror, you can't exactly correlate the meaning of an input sentence to an output sentence. With factual questions, the answer is usually static (ex. "What is x?" -> "It is (definition here)"), but in two-sentence horror, an opener like "There was a ghost" could be followed up with dozens of different finishers. When exploring the dataset, we noticed that body horror was a common genre, which could have led to examples of over-correlation (where `x` -> `body horror`).

An interesting exploration here would be re-training with a Seq2Seq architecture while using pre-trained word embeddings such as Google's Word2Vec, which has a better understanding of English than what we extracted out of two-sentence horror stories.

#### Method 3: Bidirectional and Auto-Regressive Transformers (BART)
The next step now was entering the world of transformers, a recent advancement in NLP but one that has taken the field by storm. BART utilizes a Seq2Seq-like model architecture (encoder-decoder) underneath unlike common encoder or decoder-only designs (like BERT and GPT respectively). This gives us a view of potentially how Method 2 (Seq2Seq) could have performed with better word embeddings.

BART was fine-tuned rather than trained from scratch given the sheer size of this model (using Hugging Face's 🤗 Transformers library). After the recommended 3 epochs, we received what we believe are very good results:

- Ex 1: **My parents told me not to go upstairs.** i don’t know what’s worse, the fact that i’m the only one down here, or that i can hear them screaming.
- Ex 2: **There was a knock on the door.** it was the only way i could get out of the basement.
- Ex 3: **There was a loud noise coming from the basement.** it was only when i turned on the lights that i realized the noise wasn't coming from the basement.
- Ex 4: **There was a ghost.** it was the only thing keeping me alive.

#### Method 4: GPT-Neo
We decided to try this again, but using a decoder-only architecture as it was frequently claimed to be tailored specifically for text generation and creative writing tasks (whereas BART was more open-ended). <i> Note, for this model, we experimented with full story generation rather than providing an input. </i>

- Ex 1: **While walking to school, my classmates exclaimed "Oh god, they're screaming!"** That's when I realized I should have locked my phone...
- Ex 2: **When the kidnapper made me guess where he kept my daughter, I went for the basement and he said "Correct!" allowing me to see her.** But when I found her severed head in there, I learned that every other choice would have been correct as well.
- Ex 3: **I ate my sister in the womb.** Too bad for mom I was seven at the time.
- Ex 4: **“You’re so sweet”, I told my wife as she started to rub my back.'** “What’d you say?” She shouted from the other room.

Now that the model is in charge of the full story, we can see the bias in body horror here. About half the stories generated here had some emphasis of using the body in some way to provoke the reader. We'd imagine authors find it easier to create discomfort when refering to things readers can directly experience or imagine.

## Takeaways

We found the BART model our favorite of all four methods attempted. Thus, we decided to move forward with it for future steps (mentioned below in `Moving Forward`). To make the model publicly accessible, we created a Space on Hugging Face where users can interact with it (link above). In addition, we included links to the dataset, execution of model training, and the model itself.

## Moving Forward

Currently, we have only accessed our generated stories manually. By reading them, we have a general sense of what seems like a good story or not. Our immediate next step would be to properly assess these stories in some way. Ideally, we would like to quantify and give them a score. We have tried two approaches to assessing stories with cosine and Jaccard similarity, but found neither appropriate. These would attempt to find the closest story by word positioning, but that doesn't properly convey its meaning.

We have also tried using things like BERTScore and BARTScore to find the closest story semantically, but had issues properly running these models in time (on a local system, it would take about 20 seconds to find the most similar story). Assessing these stories quantitatively is a project we will continue to work on.
