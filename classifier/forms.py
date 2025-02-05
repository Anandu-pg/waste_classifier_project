from django import forms


from .models import Booking

class DateInput(forms.DateInput):
    input_type = "date"

class BookingForm(forms.ModelForm):
    class Meta:
        model = Booking
        fields = '__all__'

        widgets = {
            'booking_date' : DateInput(),

        }

        
        labels = {
            'user_name' : "Patient Name: ",
            'user' : "Patient Phone: ",
            'p_email' : "Patient Email",
            'doc_name' : "Doctor Name",
            'bookinag_date' : "Booking Date: ",

        }