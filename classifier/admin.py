from django.contrib import admin
from .models import ContactMessage, Profile

# Register the ContactMessage model
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'created_at')
    search_fields = ('name', 'email', 'message')
    list_filter = ('created_at',)

# Register the Profile model
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'full_name', 'phone', 'address', 'email', 'created_at')
    search_fields = ('user__username', 'full_name', 'phone', 'addres', 'email')
    list_filter = ('created_at',)

# Register the models with the admin site
admin.site.register(ContactMessage, ContactMessageAdmin)
admin.site.register(Profile, ProfileAdmin)
