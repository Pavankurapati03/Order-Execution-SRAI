from django.urls import path
from . import views

urlpatterns = [
    # Root redirect
    path("", views.root_redirect, name="root_redirect"),

    # Account & License
    path("portal/open-account", views.open_account_view, name="open_account"),
    path("portal/auto-activate", views.auto_activate_view, name="auto_activate"),
    path("license", views.verify_view, name="license"),
    path("portal/verify", views.verify_view, name="verify"),
    path("portal/logout", views.logout_view, name="logout"),
    path("license/logout", views.logout_view, name="license_logout"),

    # Credits & Stripe
    path("portal/credits", views.credits_view, name="credits"),
    path("portal/credits/buy", views.credits_view, name="credits_buy"),
    path("portal/credits/success", views.stripe_success_view, name="stripe_success"),
    path("api/stripe/webhook", views.stripe_webhook_view, name="stripe_webhook"),
    path("portal/account", views.account_view, name="account"),
    path("portal/credits/ledger", views.credit_ledger_view, name="credit_ledger"),
    path("portal/credits/available", views.credit_ledger_view, name="credit_available"),
    path("portal/terms", views.terms_view, name="terms"),
    path("portal/terms/", views.terms_view, name="terms_slash"),

    # Marketplace & Dashboards
    path("portal/marketplace", views.marketplace_view, name="marketplace"),
    path("portal/order_execution", views.marketplace_view, name="order_execution_alias1"),
    path("portal/order-execution", views.marketplace_view, name="order_execution_alias2"),
    path("portal/marketplace/sales_forecasting", views.sales_forecasting_view, name="sales_forecasting"),
    path("portal/marketplace/demand_planning", views.demand_planning_view, name="demand_planning"),
    path("portal/marketplace/procurement", views.procurement_view, name="procurement"),
    path("portal/marketplace/procurement_engine", views.procurement_view, name="procurement_alias"),
    path("portal/dashboard/<str:session_id>", views.dynamic_dashboard_view, name="dynamic_dashboard"),
    path("portal/executive_dashboard", views.executive_dashboard_view, name="executive_dashboard_standalone"),
    path("portal/executive/<str:session_id>", views.executive_dashboard_view, name="executive_dashboard_session"),
    path("portal/samples/<str:dashboard_name>", views.sample_dashboard_view, name="sample_dashboard"),
    path("portal/samples/<str:dashboard_name>/", views.sample_dashboard_view, name="sample_dashboard_slash"),

    # Chatbot & APIs
    path("portal/chatbot", views.chatbot_view, name="chatbot"),
    path("app", views.chatbot_view, name="app_core"),
    path("app/core", views.chatbot_view, name="app_core_alias"),
    path("api/chat/restart", views.chat_restart_view, name="chat_restart"),
    path("api/chat/interactive", views.chat_interactive_view, name="chat_interactive"),
    path("api/summary", views.get_pipeline_summary_view, name="pipeline_summary"),
    path("api/chat/upload", views.chat_upload_view, name="chat_upload"),

    # Reports & Downloads
    path("api/eda/view/<str:session_id>", views.view_eda_report_view, name="view_eda_report"),
    path("api/eda/download-pdf/<str:session_id>", views.download_eda_pdf_view, name="download_eda_pdf"),
    path("api/master/download/<str:session_id>", views.download_master_csv_view, name="download_master_csv"),
    path("api/master/download-csv/<str:session_id>", views.download_master_csv_view, name="download_master_csv_alias"),

    # Agentic AI Decision Intelligence Engine (UNDERSTAND Stage)
    path("api/agentic/investigate/<str:session_id>", views.agentic_investigate_view, name="agentic_investigate"),
    path("api/agentic/investigate", views.agentic_investigate_view, name="agentic_investigate_default"),
    path("api/agentic/dossier/<str:session_id>", views.agentic_dossier_view, name="agentic_dossier"),
    path("api/agentic/dossier", views.agentic_dossier_view, name="agentic_dossier_default"),
    path("api/agentic/action/approve", views.agentic_action_approve_view, name="agentic_action_approve"),
    path("api/agentic/investigation_trace/run/<str:session_id>", views.agentic_investigation_trace_run_view, name="agentic_investigation_trace_run"),
    path("api/agentic/investigation_trace/run", views.agentic_investigation_trace_run_view, name="agentic_investigation_trace_run_default"),
    path("api/agentic/job_status/<str:job_id>", views.agentic_job_status_view, name="agentic_job_status"),
]
