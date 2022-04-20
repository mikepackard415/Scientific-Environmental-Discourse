#!/bin/bash

mysql --ssl-ca=../wos-db.uchicago.edu.ca \
      --ssl-key=../wos-db.uchicago.edu-key.pem \
      --ssl-cert=../wos-db.uchicago.edu-cert.pem \
      -h wos-db.uchicago.edu --user=mcpackard wos_new < wos-env.sql > wos-env.tab

git add wos-env.tab
git commit -m 'New wos data pull'
git push