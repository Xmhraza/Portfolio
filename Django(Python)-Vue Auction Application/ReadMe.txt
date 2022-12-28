This is a web application that is used as an auction application

Django (python) is used as the backend and Vite/Vue (typescript is used) is used as the frontend

Features include:

1. Creating a profile (Update a profile if you already have one)
2. Add a new item
3. View the auction site (all items that are not expired are shown here)
4. Dynamic item pages that show the description of the item and has a question and answer 5. system, and a bidding system.
6. If you are the highest bidder and the item is expired (expires at the set date), generation of an email to the given email address in your profile
7. Please note: if the send mail function is not working on a mac laptop please change the EMAIL_HOST_PASSWORD to ghbhwmqybtrgfbpn. You can find this in the settings.py file in the mysite folder
8. Answer page that has all the questions asked to the user

Styling is not implemented on the application at the time. All the features work if the user is logged in and a session id is established

How to run the application:

Please install and set up Django (See the django documentation)
You will have to install Vue/Vite and crispy forms (See the django documentation)
Once Django is set up, you will have to run two servers
To run the backend server, go to the main folder which contains the manage.py file and run the command (on a terminal) python manage.py runserver
Open another terminal, go to the FrontEnd folder (cd Frontend) and run the command
npm run dev. If there are any dependencies required, use npm install

The Frontend functionality will not be working since you are required to log in
To do this, go to localhost:8000/signup/ and create an account. You will be redirected to log in. Log in with the new account (note you will not get any confirmation messages at the time. The way to confirm that you are logged in is to check the cookies and see if the session id is created. If the frontend functionality is working, then you have successfully logged)
Now you can use the application and all of its features. I do plan on adding styling in the future

