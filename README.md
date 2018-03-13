# Portland Data Science Group: Stock Market Data  (Oct 2017/ Mar 2018)

This has my notebooks on analyzing stock data from 1970-2017.

## Oct 2017 
In the October 2017 session, I mostly focused on characterizing the data,
and looking for departures from the classic assumptions about stocks
(that they are geometric Brownian motions).  There were also some
attempts at fitting ARIMA models on the log-differenced data, and removing seasonality 
(where I think the cure was worse than the disease).  

## March 2018

In this more recent March 2018 session, I'm part of a group working at 
applying Neural networks to the same data.  This implements 
a small recurrent neural network in Tensorflow to forecast day ahead
stock prices.  The network is defined in recurrent_network.py.
This is intended as a starter so people can play with the network
(since I've found getting going with Tensorflow can be difficult).

Ideally, that would be ported over to Keras, but I already had this mostly
ready to go.
