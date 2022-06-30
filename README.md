# nonlinearLSTM
Trying to train LSTMs for non-linear models.

## read stuff
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/ done
    - really good introduction to sequential models
    - mostly focused on language models, however many general tips
- https://www.researchgate.net/publication/349681865_Deep_Learning_with_Real_Data_and_the_Chaotic_Double_Pendulum mostly done
    - pretty similar to my work, good reference for code examples
- https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca done
    - good explanation of inner workings of lstm cells
    - nice example with lstm regression
- https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
    - some general tests one can run to maybe catch some bugs in your machine learning pipeline 
    - wanted to make it a habit to unit test my stuff more, as it is essential with bigger projects
    - should help with bug fixing too

- a lot of literature for language models (predicting letter after letter) --> classification
- but most things can be applied for lstm regression as well
## Create data from classic models
- create nonlinear state space models
    - given models work (heat model/ transmission)
    - more general solution better (just plug function and parameters/initials in --> train network)
    - but mostly done
- add measuring noise at output to simulate real world data acquisition
    - done, but only necessary to add later, when no other bugs
- create training and test datasets
    - done, but not in general
    - input: 0 --> n-1, [batch size, sequence length, number of features] (batch size = number of different sequences)
    - labels: 1 --> n
    - easy to just increase/change the number of features to add inputs from ode (current, change of first value, system input, etc.)
    
## Create LSTM
- create train/test functions and data pipeline
    - test/train basics done, need to adjust some stuff for regression
- create simple LSTM with pytorch
    - done? 0,1,2,3 --> 1,2,3,4 (linear) should be really simple to learn
    - loss not going down quickly enough, even for dummy data
    - either stupid bug, or model setup/change from classification to regression not complete
- visualize results
    - TODO, after loss goes down (faster)
## Expand
- expand to more variations of classic models
- different activation functions (changes non-linearity)
- change datasets to better reflect real world (distributions of inputs)
