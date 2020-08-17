# # SHOWCASE-ONE-SHOWCASE-ONE-SHOWCASE-ONE-SHOWCASE-ONE-SHOWCASE-ONE-SHOWCASE-ONE

# # == [ BERNOULLI FUNCTION START ] =============================================
# import array
# import numpy as np
# import matplotlib.pyplot as plt

# def perform_bernoulli_trials(n, p):
#     """
#         Performing 'n' Bernoulli trials
#         Determination of the probability p (success)
#         returning the number of success finding a 'default'
#     """

#     n_success = []

#     for i in range(n):
#         # Choose random number between zero and one
#         random_number = np.random.random()
#         # If less than 'p', store a 'True' as 1 into 'n_success'-list
#         if random_number < p :
#             print('Mussel', i + 1, 'of', n, '| p:',
#                   p, '| Randomly chosen number:',  round(random_number, 5))
#             n_success.append(1)

#     print('Probability (p) of \'defaults\':', n_success)

#     return n_success
# # == [ BERNOULLI FUNCTION  END  ] =============================================
