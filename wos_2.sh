#!/bin/bash

mysql --ssl-ca=../wos-db.uchicago.edu.ca \
      --ssl-key=../wos-db.uchicago.edu-key.pem \
      --ssl-cert=../wos-db.uchicago.edu-cert.pem \
      -h wos-db.uchicago.edu --user=mcpackard wos_new < wos.sql > wos.tab
