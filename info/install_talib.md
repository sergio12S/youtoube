# Install TA-Lib 

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/home/serg/VECTOR && \
    make && \
    make install

pip install --global-option=build_ext --global-option="-L/home/serg/VECTOR/lib" TA-Lib==0.4.24
rm -R ta-lib ta-lib-0.4.0-src.tar.gz