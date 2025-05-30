---
title: Capture Hidden Trends: Use Cases for Private and Decentralized ML Training
date: 2025 May 30
description: Use Cases for Private and Decentralized ML Model Training
---

Since the beginning of this year, I’ve been exploring the intersection of cryptography and machine learning and thinking what's important work on in the long term. In my last post, I shared a technical overview of the first iteration of my new project: [Publicly Verifiable, Private & Collaborative AI Training](https://yuriko.io/posts/verifiable-federated-learning/) (for brevity, I’ll call it private & decentralized ML model training from now on).

To summarize, I prototyped a system that allows mutually distrusting parties in a decentralized protocol to collaboratively train a machine learning model, without exposing their private dataset to one another. All participants in the system use zero-knowledge proofs to verify the integrity of their local computations, including client-side training and server-side model aggregation. 

In this post, I will explore potential use cases and social implications of this technology that I’ve been reflecting on.

## Table of Contents
1. [Capture Hidden Trends](#capture-hidden-trends)  
   1.1 [Private Data Exists in Silo](#private-data-exists-in-silo)  
   1.2 [Structural Privilege in Data Collection](#structural-privilege-in-data-collection)  
   1.3 [Pull-style → Push-style Data Science](#pull-style-push-style-data-science)  
2. [Usecase 1: Crowdsourced Health Data Analysis](#usecase-1-crowdsourced-health-data-analysis)
3. [Usecase 2: Private Fine-tuning for Vulnerable Subgroups](#usecase-2-private-fine-tuning-for-vulnerable-subgroups)  
   3.1 [Tailor-made Models for Marginalized Communities](#tailor-made-models-for-marginalized-communities)  
   3.2 [Exporting Crypto Credit Score to TradFi for the Unbanked](#exporting-crypto-credit-score-to-tradfi)  
   3.3 [Model Merging for Intersectional Demographics](#model-merging-for-intersectional-demographics)  
4. [Usecase 3: Recommendation System for dApps](#usecase-3-recommendation-system-for-dapps)
5. [Usecase 4: Privacy-preserving Model Training for Decoding Biometric Data](#usecase-4-privacy-preserving-model-training-for-decoding-biometric-data)
6. [Note on Verifiability and Bonus Project Idea](#note-on-verifiability-and-bonus-project-idea)  
   6.1 [Verifiability for Malicious Adversary](#verifiability-for-malicious-adversary)  
   6.2 [Bandwidth-Efficient Edge Device Training](#bandwidth-efficient-edge-device-training)  
7. [End Note](#end-note)

## Capture Hidden Trends
Before diving into each use case idea, I want to talk about a recurring theme among them, which has shaped the direction of this project. 


### Private Data Exists in Silo
> Definition. Data point (noun): an identifiable element in a dataset.
>
> Definition. Data / Dataset (noun): facts and statistics collected together for reference or analysis.

First, **data points**, by definition **become meaningful in relation to the other data points**. Let's say I step on a scale today and I see some number. If there are no other weights (either mine or other people's) that I want to compare it to, this number alone does not tell me any insight. When individual data points are grouped together, they form a dataset — something that can be analyzed to extract patterns or insights.

Second, *some* data points exist in silos, and **only those in positions of power and those with access to sufficient infrastructure are able to collect them** (not necessarily with proper consent from the data owners but that's another point) **and form a dataset**. 

For example, imagine I wanted to compare my income to that of other female, Asian cryptography researchers living in Europe. This would be extremely difficult for the following reasons:

1. As an individual unaffiliated with any scientific institution, I have no way to directly coordinate with people in this specific demographic to collect such data.

2. Even if a global income dataset existed, filtering it by such personal attributes—female, Asian, based in Europe—would be nearly impossible due to privacy concerns.

### Structural Privilege in Data Collection
I see a **structural privilege** here. Data tends to get collected and studied when powerful institutions decide to do so, in a way that they designed. 

For those of you who're unfamiliar with the concept of structural privilege (or oppression), historically various systems in society have been designed by a dominant group in a way that serves their own interests—intentionally or unintentionally. As a result, other marginalized groups have faced implicit systemic disadvantages, often because their needs are not reflected during the design process.

Prime example includes the [history of voting rights in the US](https://data-feminism.mitpress.mit.edu/pub/vi8obxh7#nn6fq2gj2xv), but more specifically for data science, there is a category of diseases called [Neglected Tropical Diseases (NTDs)](https://www.who.int/news-room/questions-and-answers/item/neglected-tropical-diseases) which are common in low-income populations in developing countries. It affects over 1 billion people worldwide, but is under-studied with a lack of market incentive since pharma companies make little profit from treating poor populations. 

<figure style="text-align: center; margin: 2rem;">
<img src="https://hackmd.io/_uploads/r1fvoWIfxe.png)" style="margin: 0;"/>

<figcaption style="font-style: italic; margin-top: 0.5rem;">Number of people requiring treatment against neglected tropical diseases by <a href="https://ourworldindata.org/grapher/interventions-ntds-sdgs" target="_blank" rel="noopener noreferrer">Our World In Data</a>
</figcaption>

</figure>



Another example of structural disadvantage appears in automotive safety testing, where crash tests have long prioritized dummies modeled on the “average male body.” Since the 1960s when testing started, average female dummies were either absent or used in ways that ignored key anatomical differences, often justified by funding constraints. As a result, research has shown that [women face as much as 73% higher risks of fatality or serious injury in car crashes](https://news.virginia.edu/content/study-new-cars-are-safer-women-most-likely-suffer-injury) in the past. It is reasonable to infer that this systematic exclusion of women from safety design decisions is closely linked to the male-dominated nature of the automotive industry (another resource that explains the historical context is [here](https://www.consumerreports.org/car-safety/crash-test-bias-how-male-focused-testing-puts-female-drivers-at-risk/)).

<figure style="text-align: center; margin: 2rem;">
<img src="https://hackmd.io/_uploads/SkhAgW8Mel.jpg)" style="margin: 0;"/>

<figcaption style="font-style: italic; margin-top: 0.5rem;">
Managing Director Sir David Brown stands beside a damaged Aston Martin after crash testing, October 17th 1968.  
</figcaption>
</figure>

As we can see from these examples, the disparity between prioritized and ignored data stems from the combination of **pursuit for profitable research** (which isn’t limited to industry; academic research also depends on funding, often prioritizing data collection and analysis aligned with industry interests) and the **dominance of privileged group** in decision-making positions. 

### Pull-style → Push-style Data Science
What happens if we can gain more **agency** over which of *our* data we collect and how we make use of it? More precisely, what if we complemented<sup>1</sup> the conventional *pull* style data science, where institutions decide which data is worth collecting, with a *push* style, where individuals proactively contribute their data (in a privacy-preserving way, otherwise this doesn't work)? Such a shift could enable **collaborative data analysis among people who share similar interests, goals, or curiosities**.

I believe there are **many hidden patterns within private data** scattered across the world. There is an invisible trend embedded in a [missing dataset](https://mimionuoha.com/the-library-of-missing-datasets)—datasets that should, but don’t exist yet for structural oppression, but **the individuals hold this data haven’t had the means to effectively coordinate for privacy concerns**.

<figure style="text-align: center; margin: 2rem;">
<img src="https://hackmd.io/_uploads/Skwy7bIGel.jpg)" style="margin: 0;"/>

<figcaption style="font-style: italic; margin-top: 0.5rem;">
<a href="https://mimionuoha.com/the-library-of-missing-datasets-v-20" target="_blank" rel="noopener noreferrer">The Library of Missing Datasets 2.0 (2018)</a> - mixed media installation focused on blackness by Mimi Onuoha
</figcaption>
</figure>




Perhaps what society ignores, or actively hides tells more about the world than what it highlights. With decentralized & private ML model training, we can **extract these patterns without exposing the underlying data itself, and make the invisible visible, on our terms**<sup>2</sup>. 


(1: I use the word *complement* intentionally here. I don't mean to dismiss the work that institutional data scientists have done so far, nor am I trying to create a dichotomy where centralized data science is “bad” and decentralized data science is inherently “good.” However, I believe more and more individuals without formal academic training or institutional affiliation will become capable of conducting valuable experiments and data analysis. I’m curious to see what hidden truths might emerge if independent researchers with more diverse backgrounds and original perspectives are given free access to whatever datasets they get curious to study.)

(2: This type of ground-up data collection isn't a completely new initiative. Scholars have coined several terms such as  [*counterdata*](https://datasociety.net/wp-content/uploads/2024/04/Keywords_Counterdata_Olojo_04242024.pdf)—data that is collected to contest a dominant institution or ideology, to describe the concept)

## Usecase 1: Crowdsourced Health Data Analysis
This use case idea represents the theme I described in the above section pretty clearly. It enables individuals to contribute (“push”) their data in a privacy-preserving manner to uncover patterns within a specific demographic. Data contributors could verify that they belong to a target demographic (again, preserving privacy—for instance via ZK) and perform local training on their own data. This will exactly allow us to "capture hidden patterns" within private dataset, which have traditionally been difficult to collect in one place. That said, I still need to think more on whether we can assume each individual holds enough data to train a meaningful model, which depends on specific use cases. If they only hold a single data point, which is obviously insufficient to train a model alone, then contributors might instead submit their data to MPC nodes and delegate the training on a larger volume of data collected from various data contributors. That shifts the architecture closer to Hashcloak’s [noir-mpc-ml](https://github.com/hashcloak/noir-mpc-ml/tree/master) rather than my prototype based on “zk-federated-learning.” 


## Usecase 2: Private Fine-tuning for Vulnerable Subgroups

This is an idea I'm personally most excited about.
Suppose we have a pre-trained foundation model (like LLMs) out there and some blockchain nodes hold a specific dataset representing a marginalized, smaller community. This kind of dataset is difficult to collect with "pull" style, due to their sensitive attributes, such as race, gender, disability status, sexual orientation etc, as I explained in the first section. ([Guy Roghbulm](https://guyrothblum.wordpress.com/), a research scientist at Apple explains that "it can be perfectly appropriate and necessary to use sensitive features (for ML), but frustratingly,  it's sometimes difficult for legal reasons in the US" in this [lecture](https://youtu.be/iB8Qq_Ew2aA?si=VWFbyWu8GqIOm572&t=205) from [Graduate Summer School on Algorithmic Fairness](https://www.ipam.ucla.edu/programs/summer-schools/graduate-summer-school-on-algorithmic-fairness/)) So instead, what if each client with private dataset can locally fine-tune a foundation model and generalize nuanced patterns unique to this specific subgroup? Those are **patterns that are often overlooked or averaged out** in a global model trained on a vast dataset.

<figure style="text-align: center; margin: 2rem;">
<img src="https://hackmd.io/_uploads/H1BkibUflg.png" style="margin: 0;"/>

<figcaption style="font-style: italic; margin-top: 0.5rem;"><a href="https://dwork.seas.harvard.edu/" target="_blank" rel="noopener noreferrer">Cynthia Dwork </a>explains in this <a href="https://youtu.be/rtVxxSzJT3Y?si=eLinvDplO46tpzV5&t=1704" target="_blank" rel="noopener noreferrer">lecture</a> the cause of algorithmic bias such as face recognition system failing to detect black woman's face until she puts a white mask
</figcaption>

</figure>

(Note: Initially I was vaguely thinking decentralized AI training can *reduce* algorithmic bias. I still believe it could mitigate the problem, but I think "reduce" is a wrong phrasing. I would say machine learning *inherently is a technology to create bias*. It *generalizes some patterns* within a group and predicts some outcomes for unseen data points *assuming that this pattern persists*. This directly fits the definition of creating and using bias. So I would argue, the only way we can make a *fair* use of it (with a cost of more customization/less automation) is to **narrow down the scope of its usage and carefully design the training dataset accordingly**.)

### Tailor-made Models for Marginalized Communities
For example, this "narrowly scoped, tailored" model can be used in tasks such as **financial risk assessment, medical detection, and hiring for marginalized communities**. Institutions that care about creating more fairness and equal opportunities for them such as [Community Development Financial Institution (CDFIs)](https://en.wikipedia.org/wiki/Community_development_financial_institution) or [Out For Undergrad(O4U)](https://www.outforundergrad.org/) would be interested in building this tailor-made model without collecting required training data with sensitive attributes. What's even cooler is, companies and public institutions will **be able to publicly verify their design of training dataset** tailored to specific communities, so that they can prove their intention for fairness towards these groups.

### Exporting Crypto Credit Score to TradFi for the Unbanked
Another potentially impactful idea is to **privately build a credit scoring model for the unbanked, inclding their real-world sensitive attributes while keeping them private**. This model could then be exported to traditional financial institutions, signaling the patterns of real-world personal attributes for those who have been responsibly borrowing money in crypto. This would create new financial opportunities/pathways to high-street banks even for those who began with zero credit in traditional finance. 


### Model Merging for Intersectional Demographics
Additional idea: Now, I'm curious to see what happens if we merge each of these fine-tuned models and build an **intersectional model**. I suppose such a combined model would **generalize patterns in intersectional identities better than the individual models alone**. For example, in a hiring context, merging a model that identifies strong candidates from one minority group (e.g. Hispanic) with another focused on a different group (e.g. women) could improve performance for those who belong to both groups (e.g. Hispanic women). Another example is, you can also ask a question like “What’s the likelihood of *White male vegans* developing osteoporosis?” This kind of question involves overlapping personal identity factors that single-group models may not capture well.

<figure style="margin: 2rem auto; width: 60%; text-align: center">
<img src="https://hackmd.io/_uploads/Byprykvzlg.png" style="display: block; max-width: 100%"/>

<figcaption style="font-style: italic; margin-top: 0.5rem;">My mental model of intersectional models. Each circle represents fine-tuned models. Overlapped intersections represent niche communities
</figcaption>

</figure>

Following these examples, I believe model merging techniques could be extremely powerful. If we have access to models trained on private data from smaller demographic groups, we can combine them to build **custom models tailored to even more niche communities** we want to make predictions about.

On that note, one interesting method of model merge is [Evolutionary Model Merge](https://sakana.ai/evolutionary-model-merge/). What's special about this method is it automates merging models with different specialities and modalities such as Japanese-LLM with mathematical reasoning capability or Image Generation Model.

## Usecase 3: Recommendation System for Dapps
This idea may be less novel, but it's likely the most realistic usecase in my opinion. As we all know, decentralized applications (dApps) have pseudonymous/anonymous users often privacy-conscious, which makes it difficult for dApps service providers to collect personal user profiles or track their in-app behavior. This creates a challenge to build personalized recommendation systems, which traditionally depend on a large volume of personal data collection in a central server for training ML models. 
If decentralized & private model training can scale to support millions of clients or allow delegating such training to MPC nodes (which is more realistic), dApps could deliver personalized experiences without compromising user privacy. (My [attempt](https://forum.devcon.org/t/dip-65-private-recommendation-system-for-devconpassport-app/5347) of such application developement)

## Usecase 4: Privacy-preserving Model Training for Decoding Biometric Data
This idea is a bit of a jump from the others, but it was actually the initial motivation that led me to research more on private ML model training. At the end of 2024, I was introduced to the field of brain-computer interface (BCI). I learned that after capturing brain signals with whatever method (e.g. EEG or ultrasound), BCIs typically involve a “decoding” process that interprets raw brain wave data into meaningful labels such as physiological states, based on its frequency. (For example, delta waves with 0.5–4 Hz are associated with deep sleep or unconsciousness, while beta waves with 13–30 Hz are linked to alertness, active thinking.) This decoding is generally powered by machine learning model inference. With public information right now, companies seem to rely on labeled datasets collected in clinical or research environments to train these models. However, it’s reasonable to assume **they may eventually seek to collect training data directly from end users**. This could merely be my speculation, but if it actually happens, it would raise serious privacy concerns and be subject to strict regulation. (You might remember [WorldCoin was suspended](https://www.bbc.co.uk/news/world-africa-66383325) in some European/African/Asian countries for failing to demonstrate proper handling of iris data) Even in a world where “privacy doesn’t sell,” regardless of how end users would feel, it won't be easy for private companies to collect such sensitive biometric data and use it for businesses. In the near future, I believe **introducing privacy-preserving training methods to commercial companies that handle biometric data will be demanded**, enabling model improvement without forcing users to compromise their sensitive data.


## Note on Verifiability and Bonus Project Idea

### Verifiability for Malicious Adversary
I’ve been exploring additional motivations for adding verifiability to federated learning (FL), beyond the aforementioned cases of deploying FL on a decentralized network (where participants are mutually distrusting and thus require proof of correct local computations).

In cryptography world, this is a setting that demands **security against malicious adversaries**, as opposed to the semi-honest (or honest-but-curious) adversary model. (A helpful explanation can be found [here](https://crypto.stackexchange.com/questions/102283/security-against-malicious-adversaries-in-mpc)). Traditionally, federated learning has been applied in collaborations where a baseline level of trust or business alignment already exists (mostly equivalent to 'semi-honest' setting)—such as between different branches of the same bank (e.g., U.S. and European divisions), or across hospitals within the same region. In these cases, FL is often used not because the parties distrust each other, but because data sharing is restricted by regulations like GDPR. 
However, general trend in ML training is that the architectures have been shifting toward **distributed edge device training** for better scalability. Edge device training exactly fits in the definition of a setting which requires security against malicious adversaries.

### Bandwidth-Efficient Edge Device Training
And here is a new idea to gain even more efficiency utilizing verifiability: In some cases, local models trained on edge devices can reach comparable accuracy even if their parameters differ slightly. That means they may not need to synchronize with the central server to build a global model as frequently. During these “idle” periods, each edge device could instead submit a succinct proof attesting that:

1. Their model was trained correctly, and
2. The resulting accuracy remains within an acceptable bound.

This approach can significantly reduce required bandwidth and computational cost to aggregate local models on the central server, compared to transmitting full model updates each round. 

## End Note
In this post, I listed up potential use cases and project ideas for [Publicly Verifiable, Private & Collaborative AI Training](https://yuriko.io/posts/verifiable-federated-learning/).
I’d immensely appreciate feedback from experts in the relevant fields. Also I’m currently conducting this research independently and looking for organizations that can host me to further develop this work in partnership with external teams or clients. If you're interested, please reach out to: yuriko dot nishijima at google mail.


-----

Special thanks to [Shintaro](https://shibashintaro.com/), [Lucas](https://x.com/luksgrin) (you can check his [commentary](https://hackmd.io/@jdsIUqinSz2KRzqrrqU9Ig/SyE5qp2bex) for this post with his molecular biology research background), and [Octavio](https://gitlab.com/OctavioDuarte/) for valuable feedback and insightful discussions.

If you have any feedback or comments on this post and are willing to engage in a meaningful discussion, please leave them in the HackMD draft: https://hackmd.io/@yuriko/Bk28WMRxgl 
