class iirc(object):

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit(self, ratings_mat):
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()


    def _set_neighborhoods(self):
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    
    def pred_user(self, user_id, movie_id):
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        relevant_items = np.intersect1d(self.neighborhoods[movie_id],
                                            items_rated_by_this_user,
                                            assume_unique=True)  
        out = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[movie_id, relevant_items] / \
                self.item_sim_mat[movie_id, relevant_items].sum()
        cleaned_out = np.nan_to_num(out)
        return cleaned_out
    
    def rec_anime(self, movie_id):
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        relevant_items = np.intersect1d(self.neighborhoods[movie_id],
                                            items_rated_by_this_user,
                                            assume_unique=True)  
        out = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[movie_id, relevant_items] / \
                self.item_sim_mat[movie_id, relevant_items].sum()
        cleaned_out = np.nan_to_num(out)
        return cleaned_out