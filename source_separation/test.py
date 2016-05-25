#!/usr/bin/python

def train_srs():
    # Load train set    


    # Create net
    rnn = neural_network.RNN(INPUT_SIZE, 2, noisy, 2, loss=source_separation_loss_function)

    # Train net
    rnn.fit(noisy, target)

    # Save net
    rnn.save()
    
    # Result
    print 'LOL'


def test_srs():
    # Load test set
    X = 2
    # Load net

    # Test net

    # Save output result

    # Result
