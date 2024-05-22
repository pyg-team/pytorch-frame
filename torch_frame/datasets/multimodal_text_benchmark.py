from __future__ import annotations

import os.path as osp

import numpy as np
import pandas as pd

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.utils.split import SPLIT_TO_NUM


class MultimodalTextBenchmark(torch_frame.data.Dataset):
    r"""The tabular data with text columns benchmark datasets used by
    `"Benchmarking Multimodal AutoML for Tabular Data with Text Fields"
    <https://arxiv.org/abs/2111.02705>`_. Some regression datasets' target
    column is transformed from log scale to original scale.

    Args:
        name (str): The name of the dataset to download.
        text_stype (torch_frame.stype): Text stype to use for text columns
            in the dataset. (default: :obj:`torch_frame.text_embedded`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10 10 20 10
        :header-rows: 1

        * - Name
          - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #cols (text)
          - #cols (other)
          - #classes
          - Task
          - Missing value ratio
        * - product_sentiment_machine_hack
          - 6,364
          - 0
          - 1
          - 1
          - 0
          - 4
          - multiclass_classification
          - 0.0%
        * - jigsaw_unintended_bias100K
          - 125,000
          - 29
          - 0
          - 1
          - 0
          - 2
          - binary_classification
          - 41.4%
        * - news_channel
          - 25,355
          - 14
          - 0
          - 1
          - 0
          - 6
          - multiclass_classification
          - 0.0%
        * - wine_reviews
          - 105,154
          - 2
          - 2
          - 1
          - 0
          - 30
          - multiclass_classification
          - 1.0%
        * - data_scientist_salary
          - 19,802
          - 0
          - 3
          - 2
          - 1
          - 6
          - multiclass_classification
          - 12.3%
        * - melbourne_airbnb
          - 22,895
          - 26
          - 47
          - 13
          - 3
          - 10
          - multiclass_classification
          - 9.6%
        * - imdb_genre_prediction
          - 1,000
          - 7
          - 1
          - 2
          - 1
          - 2
          - binary_classification
          - 0.0%
        * - kick_starter_funding
          - 108,128
          - 1
          - 3
          - 3
          - 2
          - 2
          - binary_classification
          - 0.0%
        * - fake_job_postings2
          - 15,907
          - 0
          - 3
          - 2
          - 0
          - 2
          - binary_classification
          - 23.8%
        * - google_qa_answer_type_reason_explanation
          - 6,079
          - 0
          - 1
          - 3
          - 0
          - 1
          - regression
          - 0.0%
        * - google_qa_question_type_reason_explanation
          - 6,079
          - 0
          - 1
          - 3
          - 0
          - 1
          - regression
          - 0.0%
        * - bookprice_prediction
          - 6,237
          - 2
          - 3
          - 3
          - 0
          - 1
          - regression
          - 1.7%
        * - jc_penney_products
          - 13,575
          - 2
          - 1
          - 2
          - 0
          - 1
          - regression
          - 13.7%
        * - women_clothing_review
          - 23,486
          - 1
          - 3
          - 2
          - 0
          - 1
          - regression
          - 1.8%
        * - news_popularity2
          - 30,009
          - 3
          - 0
          - 1
          - 0
          - 1
          - regression
          - 0.0%
        * - ae_price_prediction
          - 28,328
          - 2
          - 5
          - 1
          - 3
          - 1
          - regression
          - 6.1%
        * - california_house_price
          - 47,439
          - 18
          - 8
          - 2
          - 11
          - 1
          - regression
          - 13.8%
        * - mercari_price_suggestion100K
          - 125,000
          - 0
          - 6
          - 2
          - 1
          - 1
          - regression
          - 3.4%
    """
    base_url = 'https://automl-mm-bench.s3.amazonaws.com'

    classification_datasets = {
        'product_sentiment_machine_hack', 'data_scientist_salary',
        'melbourne_airbnb', 'news_channel', 'wine_reviews',
        'imdb_genre_prediction', 'fake_job_postings2', 'kick_starter_funding',
        'jigsaw_unintended_bias100K'
    }

    regression_datasets = {
        'google_qa_answer_type_reason_explanation',
        'google_qa_question_type_reason_explanation', 'bookprice_prediction',
        'jc_penney_products', 'women_clothing_review', 'ae_price_prediction',
        'news_popularity2', 'california_house_price',
        'mercari_price_suggestion100K'
    }

    _csv_datasets = {
        'product_sentiment_machine_hack', 'data_scientist_salary',
        'news_channel', 'wine_reviews', 'imdb_genre_prediction',
        'fake_job_postings2', 'kick_starter_funding', 'bookprice_prediction',
        'jc_penney_products', 'news_popularity2', 'california_house_price'
    }

    _dataset_url_map = {
        'product_sentiment_machine_hack': 'machine_hack_product_sentiment',
        'data_scientist_salary':
        'machine_hack_competitions/predict_the_data_scientists_salary_in_india_hackathon/',  # noqa
        'melbourne_airbnb': 'airbnb_melbourne',
        'news_channel': 'news_channel',
        'wine_reviews': 'wine_reviews',
        'imdb_genre_prediction': 'imdb_genre_prediction',
        'fake_job_postings2': 'fake_job_postings2',
        'kick_starter_funding': 'kick_starter_funding',
        'jigsaw_unintended_bias100K': 'jigsaw_unintended_bias100K',
        'google_qa_answer_type_reason_explanation': 'google_quest_qa',
        'google_qa_question_type_reason_explanation': 'google_quest_qa',
        'bookprice_prediction':
        'machine_hack_competitions/predict_the_price_of_books/',
        'jc_penney_products': 'jc_penney_products',
        'women_clothing_review': 'women_clothing_review',
        'ae_price_prediction': 'ae_price_prediction',
        'news_popularity2': 'news_popularity2',
        'california_house_price': 'kaggle-california-house-prices',
        'mercari_price_suggestion100K': 'mercari_price_suggestion100K'
    }

    _dataset_target_col = {
        'product_sentiment_machine_hack': 'Sentiment',
        'data_scientist_salary': 'salary',
        'melbourne_airbnb': 'price_label',
        'news_channel': 'channel',
        'wine_reviews': 'variety',
        'imdb_genre_prediction': 'Genre_is_Drama',
        'fake_job_postings2': 'fraudulent',
        'kick_starter_funding': 'final_status',
        'jigsaw_unintended_bias100K': 'target',
        'google_qa_answer_type_reason_explanation':
        'answer_type_reason_explanation',
        'google_qa_question_type_reason_explanation':
        'question_type_reason_explanation',
        'bookprice_prediction': 'Price',
        'jc_penney_products': 'sale_price',
        'women_clothing_review': 'Rating',
        'ae_price_prediction': 'price',
        'news_popularity2': 'log_shares',
        'california_house_price': 'Sold Price',
        'mercari_price_suggestion100K': 'log_price'  # Post process?
    }

    _dataset_splits = {
        'product_sentiment_machine_hack': ['train', 'dev'],
        'data_scientist_salary': ['train', 'test'],
        'melbourne_airbnb': ['train', 'test'],
        'news_channel': ['train', 'test'],
        'wine_reviews': ['train', 'test'],
        'imdb_genre_prediction': ['train', 'test'],
        'fake_job_postings2': ['train', 'test'],
        'kick_starter_funding': ['train', 'test'],
        'jigsaw_unintended_bias100K': ['train', 'test'],
        'google_qa_answer_type_reason_explanation': ['train', 'dev'],
        'google_qa_question_type_reason_explanation': ['train', 'dev'],
        'bookprice_prediction': ['train', 'test'],
        'jc_penney_products': ['train', 'test'],
        'women_clothing_review': ['train', 'test'],
        'ae_price_prediction': ['train', 'test'],
        'news_popularity2': ['train', 'test'],
        'california_house_price': ['train', 'test'],
        'mercari_price_suggestion100K': ['train', 'test'],
    }

    _dataset_col_to_sep = {
        'data_scientist_salary': ', ',
        'imdb_genre_prediction': ', ',
        'california_house_price': ', ',
        'melbourne_airbnb': ', ',
        'ae_price_prediction': ', ',
        'kick_starter_funding': ', ',
        'mercari_price_suggestion100K': '/',
    }

    _dataset_stype_to_col = {
        'product_sentiment_machine_hack': {
            torch_frame.text_embedded: ['Product_Description'],
            torch_frame.categorical: ['Product_Type', 'Sentiment'],
        },
        'jigsaw_unintended_bias100K': {
            torch_frame.numerical: [
                'asian', 'atheist', 'bisexual', 'black', 'buddhist',
                'christian', 'female', 'heterosexual', 'hindu',
                'homosexual_gay_or_lesbian',
                'intellectual_or_learning_disability', 'jewish', 'latino',
                'male', 'muslim', 'other_disability', 'other_gender',
                'other_race_or_ethnicity', 'other_religion',
                'other_sexual_orientation', 'physical_disability',
                'psychiatric_or_mental_illness', 'transgender', 'white',
                'funny', 'wow', 'sad', 'likes', 'disagree'
            ],
            torch_frame.text_embedded: ['comment_text'],
            torch_frame.categorical: ['target'],
        },
        'data_scientist_salary': {
            torch_frame.categorical:
            ['experience', 'job_type', 'salary', 'location'],
            torch_frame.text_embedded: ['job_description', 'job_desig'],
            torch_frame.multicategorical: ['key_skills'],
        },
        'melbourne_airbnb': {
            torch_frame.text_embedded: [
                'name', 'summary', 'space', 'description',
                'neighborhood_overview', 'notes', 'transit', 'access',
                'interaction', 'house_rules', 'host_about', 'first_review',
                'last_review'
            ],
            torch_frame.categorical: [
                'host_location', 'host_response_time', 'host_response_rate',
                'host_is_superhost', 'host_neighborhood',
                'host_has_profile_pic', 'host_identity_verified', 'street',
                'neighborhood', 'city', 'suburb', 'state', 'zipcode',
                'smart_location', 'country_code', 'country',
                'is_location_exact', 'property_type', 'room_type', 'bed_type',
                'calendar_updated', 'has_availability', 'requires_license',
                'license', 'instant_bookable', 'cancellation_policy',
                'require_guest_profile_picture',
                'require_guest_phone_verification', 'price_label',
                'host_verifications_jumio', 'host_verifications_government_id',
                'host_verifications_kba', 'host_verifications_zhima_selfie',
                'host_verifications_facebook', 'host_verifications_work_email',
                'host_verifications_google', 'host_verifications_sesame',
                'host_verifications_manual_online',
                'host_verifications_manual_offline',
                'host_verifications_offline_government_id',
                'host_verifications_selfie', 'host_verifications_reviews',
                'host_verifications_identity_manual',
                'host_verifications_sesame_offline',
                'host_verifications_weibo', 'host_verifications_email',
                'host_verifications_sent_id', 'host_verifications_phone'
            ],
            torch_frame.numerical: [
                'latitude', 'longitude', 'accommodates', 'bathrooms',
                'bedrooms', 'beds', 'security_deposit', 'cleaning_fee',
                'guests_included', 'extra_people', 'minimum_nights',
                'maximum_nights', 'availability_30', 'availability_60',
                'availability_90', 'availability_365', 'number_of_reviews',
                'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin',
                'review_scores_communication', 'review_scores_location',
                'review_scores_value', 'calculated_host_listings_count',
                'reviews_per_month'
            ],
            torch_frame.multicategorical: ['host_verifications', 'amenities'],
            torch_frame.timestamp: ['host_since'],
        },
        'news_channel': {
            torch_frame.numerical: [
                ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words',
                ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs',
                ' num_imgs', ' num_videos', ' average_token_length',
                ' num_keywords', ' global_subjectivity',
                ' global_sentiment_polarity', ' rate_positive_words',
                ' rate_negative_words'
            ],
            torch_frame.text_embedded: ['article_title'],
            torch_frame.categorical: ['channel'],
        },
        'wine_reviews': {
            torch_frame.text_embedded: ['description'],
            torch_frame.categorical: ['country', 'province', 'variety'],
            torch_frame.numerical: ['points', 'price'],
        },
        'imdb_genre_prediction': {
            torch_frame.numerical: [
                'Rank', 'Year', 'Runtime (Minutes)', 'Rating', 'Votes',
                'Revenue (Millions)', 'Metascore'
            ],
            torch_frame.categorical: ['Director', 'Genre_is_Drama'],
            torch_frame.text_embedded: ['Title', 'Description'],
            torch_frame.multicategorical: ['Actors'],
        },
        'fake_job_postings2': {
            torch_frame.text_embedded: ['title', 'description'],
            torch_frame.categorical: [
                'salary_range', 'required_experience', 'required_education',
                'fraudulent'
            ]
        },
        'kick_starter_funding': {
            torch_frame.text_embedded: ['name', 'desc', 'keywords'],
            torch_frame.categorical:
            ['disable_communication', 'country', 'currency', 'final_status'],
            torch_frame.numerical: ['goal'],
            torch_frame.timestamp: ['deadline', 'created_at'],
        },
        'google_qa_answer_type_reason_explanation': {
            torch_frame.text_embedded:
            ['question_title', 'question_body', 'answer'],
            torch_frame.categorical: ['category'],
            torch_frame.numerical: ['answer_type_reason_explanation'],
        },
        'google_qa_question_type_reason_explanation': {
            torch_frame.text_embedded:
            ['question_title', 'question_body', 'answer'],
            torch_frame.categorical: ['category'],
            torch_frame.numerical: ['question_type_reason_explanation'],
        },
        'bookprice_prediction': {
            torch_frame.text_embedded: ['Title', 'Edition', 'Synopsis'],
            torch_frame.numerical: ['Price', 'Reviews', 'Ratings'],
            torch_frame.categorical: ['Author', 'Genre', 'BookCategory'],
        },
        'jc_penney_products': {
            torch_frame.text_embedded: ['name_title', 'description'],
            torch_frame.numerical:
            ['sale_price', 'average_product_rating', 'total_number_reviews'],
            torch_frame.categorical: ['brand'],
        },
        'women_clothing_review': {
            torch_frame.text_embedded: ['Title', 'Review Text'],
            torch_frame.numerical: ['Age', 'Rating'],
            torch_frame.categorical:
            ['Division Name', 'Department Name', 'Class Name'],
        },
        'ae_price_prediction': {
            torch_frame.text_embedded: ['description'],
            torch_frame.numerical: ['price', 'rating', 'review_count'],
            torch_frame.categorical: [
                'product_name', 'brand_name', 'product_category', 'retailer',
                'color'
            ],
            torch_frame.multicategorical:
            ['style_attributes', 'total_sizes', 'available_size']
        },
        'news_popularity2': {
            torch_frame.text_embedded: ['article_title'],
            torch_frame.numerical: [
                ' n_tokens_content', ' average_token_length', ' num_keywords',
                'log_shares'
            ]
        },
        'california_house_price': {
            torch_frame.text_embedded: ['Address', 'Summary'],
            torch_frame.numerical: [
                'Sold Price', 'Year built', 'Lot', 'Bedrooms', 'Bathrooms',
                'Full bathrooms', 'Total interior livable area',
                'Total spaces', 'Garage spaces', 'Elementary School Score',
                'Elementary School Distance', 'Middle School Score',
                'Middle School Distance', 'High School Score',
                'High School Distance', 'Tax assessed value',
                'Annual tax amount', 'Listed Price', 'Last Sold Price'
            ],
            torch_frame.categorical: [
                'Type', 'Region', 'Elementary School', 'Middle School',
                'High School', 'City', 'Zip', 'State'
            ],
            torch_frame.multicategorical: [
                'Heating',
                'Cooling',
                'Parking',
                'Flooring',
                'Heating features',
                'Cooling features',
                'Appliances included',
                'Laundry features',
                'Parking features',
            ],
            torch_frame.timestamp: ['Listed On', 'Last Sold On'],
        },
        'mercari_price_suggestion100K': {
            torch_frame.text_embedded: ['name', 'item_description'],
            torch_frame.numerical: ['log_price'],
            torch_frame.categorical: [
                'item_condition_id', 'brand_name', 'shipping', 'cat1', 'cat2',
                'cat3'
            ],
            torch_frame.multicategorical: ['category_name'],
        },
    }

    def _pre_transform(self, df: pd.DataFrame,
                       target_col: str) -> pd.DataFrame:
        if self.name == 'kick_starter_funding':
            df['keywords'] = [
                item.replace('-', ' ') for item in df['keywords']
            ]
        # Post transform some regression datasets' target column
        # by transforming from log scale to original scale
        elif self.name == 'bookprice_prediction':
            df[target_col] = np.power(10, df[target_col]) - 1
            df[df[target_col] < 0][target_col] = 0
        elif self.name == 'california_house_price':
            df[target_col] = np.exp(df[target_col])
            df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
        elif self.name == 'mercari_price_suggestion100K':
            df[target_col] = np.exp(df[target_col]) - 1
            df[df[target_col] < 0][target_col] = 0
        return df

    def __init__(
        self,
        root: str,
        name: str,
        text_stype: torch_frame.stype = torch_frame.text_embedded,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
    ):
        assert name in self.classification_datasets | self.regression_datasets
        self.root = root
        self.name = name
        if not text_stype.is_text_stype:
            raise ValueError(f"`text_stype` should be a text stype, "
                             f"got {text_stype}.")
        self.text_stype = text_stype

        extension = '.csv' if name in self._csv_datasets else '.pq'
        url_name = self._dataset_url_map[self.name]

        dfs = []
        for split in self._dataset_splits[self.name]:
            path = self.download_url(
                osp.join(self.base_url, url_name, split + extension), root,
                filename=f'{self.name}_{split}{extension}')
            if extension == '.csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
            dfs.append(df)

        splits = ['train', 'val', 'test'] if len(
            self._dataset_splits[self.name]) == 3 else ['train', 'test']
        for split_df, split in zip(dfs, splits):
            split_df['split'] = SPLIT_TO_NUM[split]

        df = pd.concat(dfs, ignore_index=True)

        target_col = self._dataset_target_col[self.name]

        stype_to_col = self._dataset_stype_to_col[name]

        col_to_stype = {}
        for stype in stype_to_col:
            cols = stype_to_col[stype]
            for col in cols:
                if stype == torch_frame.text_embedded:
                    col_to_stype[col] = self.text_stype
                else:
                    col_to_stype[col] = stype

        df = self._pre_transform(df=df, target_col=target_col)
        col_to_sep = self._dataset_col_to_sep.get(name, '')

        super().__init__(df, col_to_stype, target_col=target_col,
                         split_col='split', col_to_sep=col_to_sep,
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                         col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg)
