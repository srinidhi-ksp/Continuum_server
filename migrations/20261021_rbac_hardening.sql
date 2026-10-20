-- Apply after the workspace's core, Batch 1, migration v2, AI scenario,
-- and notification migrations. This replaces prior permissive RLS policies.
BEGIN;

CREATE OR REPLACE FUNCTION public.is_active_caregiver_for(p_patient_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.caregiver_pat cp
    WHERE cp.caregiver_id = (SELECT auth.uid())
      AND cp.patient_id = p_patient_id
      AND cp.removed_at IS NULL
      AND cp.status = 'active'
  );
$$;

CREATE OR REPLACE FUNCTION public.is_primary_caregiver_for(p_patient_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.caregiver_pat cp
    WHERE cp.caregiver_id = (SELECT auth.uid())
      AND cp.patient_id = p_patient_id
      AND cp.removed_at IS NULL
      AND cp.status = 'active'
      AND cp.role = 'primary'
  );
$$;

CREATE OR REPLACE FUNCTION public.can_record_care_for(p_patient_id uuid, p_type text)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.caregiver_pat cp
    WHERE cp.caregiver_id = (SELECT auth.uid())
      AND cp.patient_id = p_patient_id
      AND cp.removed_at IS NULL
      AND cp.status = 'active'
      AND (
        cp.role = 'primary'
        OR EXISTS (
          SELECT 1
          FROM unnest(COALESCE(cp.assigned_responsibilities, ARRAY[]::text[])) AS assigned(value)
          WHERE lower(trim(assigned.value)) IN (
            'all',
            CASE lower(COALESCE(p_type, ''))
              WHEN 'meal' THEN 'meals'
              WHEN 'food' THEN 'meals'
              WHEN 'water' THEN 'daily care'
              WHEN 'sleep' THEN 'sleep'
              WHEN 'medication' THEN 'medication'
              WHEN 'symptom' THEN 'symptoms'
              WHEN 'symptoms' THEN 'symptoms'
              WHEN 'mood' THEN 'other'
              WHEN 'note' THEN 'other'
              ELSE 'daily care'
            END
          )
        )
      )
  );
$$;

REVOKE ALL ON FUNCTION public.is_active_caregiver_for(uuid) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.is_primary_caregiver_for(uuid) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.can_record_care_for(uuid, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_active_caregiver_for(uuid) TO authenticated;
GRANT EXECUTE ON FUNCTION public.is_primary_caregiver_for(uuid) TO authenticated;
GRANT EXECUTE ON FUNCTION public.can_record_care_for(uuid, text) TO authenticated;

-- Drop legacy permissive policies so PostgreSQL's OR-combined policies cannot
-- leave a bypass in place.
DO $drop_policies$
DECLARE
  policy_row record;
BEGIN
  FOR policy_row IN
    SELECT schemaname, tablename, policyname
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = ANY (ARRAY[
        'patient', 'caregiver_pat', 'care_circle_invite', 'medication',
        'medication_schedule', 'medication_log', 'daily_care_log', 'time_event',
        'scenario_card', 'scenario_option', 'scenario_attempt',
        'coordination_metric', 'emergency_contact', 'caregiver', 'care_notification'
      ])
  LOOP
    EXECUTE format(
      'DROP POLICY %I ON %I.%I',
      policy_row.policyname,
      policy_row.schemaname,
      policy_row.tablename
    );
  END LOOP;
END;
$drop_policies$;

ALTER TABLE public.patient ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.caregiver_pat ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.care_circle_invite ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medication ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medication_schedule ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medication_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_care_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.time_event ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scenario_card ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scenario_option ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scenario_attempt ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.coordination_metric ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emergency_contact ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.caregiver ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.care_notification ENABLE ROW LEVEL SECURITY;

-- A patient insert creates its primary membership in the same transaction.
CREATE OR REPLACE FUNCTION public.link_new_patient_to_creator()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'An authenticated caregiver is required to create a patient';
  END IF;

  INSERT INTO public.caregiver_pat (
    caregiver_id, patient_id, role, joining_at, relationship, status,
    alert_preference, alert_priority, assigned_responsibilities, removed_at
  )
  VALUES (
    auth.uid(), NEW.patient_id, 'primary', now(), 'Primary Caregiver', 'active',
    'all', 1, ARRAY[]::text[], NULL
  );
  RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS patient_creator_membership ON public.patient;
CREATE TRIGGER patient_creator_membership
  AFTER INSERT ON public.patient
  FOR EACH ROW EXECUTE FUNCTION public.link_new_patient_to_creator();

CREATE POLICY patient_select_active_member ON public.patient
  FOR SELECT TO authenticated
  USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY patient_insert_authenticated ON public.patient
  FOR INSERT TO authenticated
  WITH CHECK ((SELECT auth.uid()) IS NOT NULL);
CREATE POLICY patient_update_primary ON public.patient
  FOR UPDATE TO authenticated
  USING (public.is_primary_caregiver_for(patient_id))
  WITH CHECK (public.is_primary_caregiver_for(patient_id));

REVOKE INSERT, UPDATE, DELETE ON public.patient FROM authenticated;
GRANT INSERT (full_name, dob, gender, condition_type, blood_group, allergies, phone, medical_notes)
  ON public.patient TO authenticated;
GRANT UPDATE (full_name, dob, gender, condition_type, blood_group, allergies, phone, medical_notes)
  ON public.patient TO authenticated;
DO $patient_timezone_grant$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'patient' AND column_name = 'time_zone'
  ) THEN
    EXECUTE 'GRANT INSERT (time_zone) ON public.patient TO authenticated';
    EXECUTE 'GRANT UPDATE (time_zone) ON public.patient TO authenticated';
  END IF;
END;
$patient_timezone_grant$;

CREATE POLICY caregiver_pat_select_active_circle ON public.caregiver_pat
  FOR SELECT TO authenticated
  USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY caregiver_pat_update_primary_member ON public.caregiver_pat
  FOR UPDATE TO authenticated
  USING (
    public.is_primary_caregiver_for(patient_id)
    AND caregiver_id <> (SELECT auth.uid())
    AND role <> 'primary'
  )
  WITH CHECK (
    public.is_primary_caregiver_for(patient_id)
    AND caregiver_id <> (SELECT auth.uid())
    AND role IN ('family', 'secondary', 'external')
  );
CREATE POLICY caregiver_pat_self_leave ON public.caregiver_pat
  FOR UPDATE TO authenticated
  USING (
    caregiver_id = (SELECT auth.uid())
    AND removed_at IS NULL
    AND role <> 'primary'
  )
  WITH CHECK (
    caregiver_id = (SELECT auth.uid())
    AND removed_at IS NOT NULL
    AND role <> 'primary'
  );

REVOKE INSERT, UPDATE, DELETE ON public.caregiver_pat FROM authenticated;
GRANT UPDATE (
  role, relationship, status, alert_preference, alert_priority,
  assigned_responsibilities, removed_at
) ON public.caregiver_pat TO authenticated;

CREATE OR REPLACE FUNCTION public.guard_caregiver_membership_update()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
BEGIN
  IF NEW.caregiver_id IS DISTINCT FROM OLD.caregiver_id
     OR NEW.patient_id IS DISTINCT FROM OLD.patient_id THEN
    RAISE EXCEPTION 'Caregiver and patient ownership cannot be changed';
  END IF;
  IF NEW.role = 'primary' AND OLD.role <> 'primary' THEN
    RAISE EXCEPTION 'Primary caregiver role cannot be self-assigned';
  END IF;
  IF current_setting('continuum.invite_accept', true) = 'true' THEN
    RETURN NEW;
  END IF;
  IF auth.uid() IS NULL THEN
    RETURN NEW;
  END IF;
  IF auth.uid() = OLD.caregiver_id THEN
    IF OLD.role = 'primary'
       OR NEW.removed_at IS NULL
       OR NEW.role IS DISTINCT FROM OLD.role
       OR NEW.relationship IS DISTINCT FROM OLD.relationship
       OR NEW.status IS DISTINCT FROM OLD.status
       OR NEW.alert_preference IS DISTINCT FROM OLD.alert_preference
       OR NEW.alert_priority IS DISTINCT FROM OLD.alert_priority
       OR NEW.assigned_responsibilities IS DISTINCT FROM OLD.assigned_responsibilities THEN
      RAISE EXCEPTION 'Caregivers may only leave their own non-primary membership';
    END IF;
  ELSIF NOT public.is_primary_caregiver_for(OLD.patient_id) THEN
    RAISE EXCEPTION 'Only the primary caregiver may manage memberships';
  END IF;
  RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS caregiver_membership_update_guard ON public.caregiver_pat;
CREATE TRIGGER caregiver_membership_update_guard
  BEFORE UPDATE ON public.caregiver_pat
  FOR EACH ROW EXECUTE FUNCTION public.guard_caregiver_membership_update();

CREATE POLICY care_circle_invite_select_primary ON public.care_circle_invite
  FOR SELECT TO authenticated
  USING (public.is_primary_caregiver_for(patient_id));
CREATE POLICY care_circle_invite_insert_primary ON public.care_circle_invite
  FOR INSERT TO authenticated
  WITH CHECK (
    invited_by = (SELECT auth.uid())
    AND public.is_primary_caregiver_for(patient_id)
    AND role IN ('family', 'secondary', 'external')
  );
REVOKE UPDATE, DELETE ON public.care_circle_invite FROM authenticated;
REVOKE INSERT ON public.care_circle_invite FROM authenticated;
GRANT INSERT (patient_id, invited_by, email, phone, relationship, role, token)
  ON public.care_circle_invite TO authenticated;

CREATE OR REPLACE FUNCTION public.lookup_care_circle_invitation(p_token text)
RETURNS TABLE (
  patient_id uuid,
  patient_name text,
  inviter_name text,
  relationship text,
  role text,
  status text,
  expires_at timestamptz,
  is_valid boolean,
  invalid_reason text
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
  SELECT
    i.patient_id,
    p.full_name,
    c.full_name,
    i.relationship,
    i.role,
    i.status,
    i.expires_at,
    i.status = 'pending'
      AND i.expires_at > now()
      AND (i.email IS NULL OR lower(i.email) = lower(auth.jwt() ->> 'email'))
      AND (i.phone IS NULL OR i.phone = auth.jwt() ->> 'phone'),
    CASE
      WHEN i.status <> 'pending' THEN 'used'
      WHEN i.expires_at <= now() THEN 'expired'
      WHEN (i.email IS NOT NULL AND lower(i.email) <> lower(auth.jwt() ->> 'email'))
        OR (i.phone IS NOT NULL AND i.phone <> auth.jwt() ->> 'phone') THEN 'recipient_mismatch'
      ELSE NULL
    END
  FROM public.care_circle_invite i
  JOIN public.patient p ON p.patient_id = i.patient_id
  JOIN public.caregiver c ON c.caregiver_id = i.invited_by
  WHERE i.token = p_token
    AND (i.email IS NULL OR lower(i.email) = lower(auth.jwt() ->> 'email'))
    AND (i.phone IS NULL OR i.phone = auth.jwt() ->> 'phone');
$function$;

CREATE OR REPLACE FUNCTION public.accept_care_circle_invitation(p_token text)
RETURNS TABLE (patient_id uuid, patient_name text, role text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
DECLARE
  invitation public.care_circle_invite%ROWTYPE;
  patient_display_name text;
  existing_link public.caregiver_pat%ROWTYPE;
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'Authentication is required';
  END IF;

  SELECT i.* INTO invitation
  FROM public.care_circle_invite i
  WHERE i.token = p_token
    AND i.status = 'pending'
    AND i.expires_at > now()
    AND (i.email IS NULL OR lower(i.email) = lower(auth.jwt() ->> 'email'))
    AND (i.phone IS NULL OR i.phone = auth.jwt() ->> 'phone')
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Invitation is invalid, expired, or belongs to another account';
  END IF;

  PERFORM 1 FROM public.patient p
  WHERE p.patient_id = invitation.patient_id
  FOR UPDATE;
  SELECT p.full_name INTO patient_display_name
  FROM public.patient p WHERE p.patient_id = invitation.patient_id;

  IF NOT EXISTS (
    SELECT 1 FROM public.caregiver_pat cp
    WHERE cp.patient_id = invitation.patient_id
      AND cp.caregiver_id = invitation.invited_by
      AND cp.role = 'primary'
      AND cp.status = 'active'
      AND cp.removed_at IS NULL
  ) THEN
    RAISE EXCEPTION 'The invitation is no longer authorized by an active primary caregiver';
  END IF;
  IF invitation.role NOT IN ('family', 'secondary', 'external') THEN
    RAISE EXCEPTION 'Invitation has an invalid caregiver role';
  END IF;

  SELECT cp.* INTO existing_link
  FROM public.caregiver_pat cp
  WHERE cp.patient_id = invitation.patient_id
    AND cp.caregiver_id = auth.uid()
  FOR UPDATE;

  IF FOUND THEN
    IF existing_link.role = 'primary' THEN
      RAISE EXCEPTION 'An existing primary caregiver membership cannot be replaced';
    END IF;
    IF existing_link.removed_at IS NOT NULL OR existing_link.status IS DISTINCT FROM 'active' THEN
      PERFORM set_config('continuum.invite_accept', 'true', true);
      UPDATE public.caregiver_pat cp
      SET role = invitation.role,
          relationship = invitation.relationship,
          status = 'active',
          removed_at = NULL,
          assigned_responsibilities = ARRAY[]::text[]
      WHERE cp.patient_id = invitation.patient_id
        AND cp.caregiver_id = auth.uid();
    END IF;
  ELSE
    INSERT INTO public.caregiver_pat (
      caregiver_id, patient_id, role, joining_at, relationship, status,
      alert_preference, alert_priority, assigned_responsibilities, removed_at
    )
    VALUES (
      auth.uid(), invitation.patient_id, invitation.role, now(),
      invitation.relationship, 'active', 'all', 2, ARRAY[]::text[], NULL
    );
  END IF;

  UPDATE public.care_circle_invite
  SET status = 'accepted'
  WHERE invite_id = invitation.invite_id;

  RETURN QUERY SELECT invitation.patient_id, patient_display_name, invitation.role;
END;
$function$;

REVOKE ALL ON FUNCTION public.lookup_care_circle_invitation(text) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.accept_care_circle_invitation(text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.lookup_care_circle_invitation(text) TO authenticated;
GRANT EXECUTE ON FUNCTION public.accept_care_circle_invitation(text) TO authenticated;

CREATE POLICY medication_select_active ON public.medication
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY medication_insert_primary ON public.medication
  FOR INSERT TO authenticated WITH CHECK (public.is_primary_caregiver_for(patient_id));
CREATE POLICY medication_update_primary ON public.medication
  FOR UPDATE TO authenticated
  USING (public.is_primary_caregiver_for(patient_id))
  WITH CHECK (public.is_primary_caregiver_for(patient_id));
CREATE POLICY medication_delete_primary ON public.medication
  FOR DELETE TO authenticated USING (public.is_primary_caregiver_for(patient_id));

CREATE POLICY medication_schedule_select_active ON public.medication_schedule
  FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.medication m
    WHERE m.medication_id = medication_schedule.medication_id
      AND public.is_active_caregiver_for(m.patient_id)
  ));
CREATE POLICY medication_schedule_insert_primary ON public.medication_schedule
  FOR INSERT TO authenticated
  WITH CHECK (EXISTS (
    SELECT 1 FROM public.medication m
    WHERE m.medication_id = medication_schedule.medication_id
      AND public.is_primary_caregiver_for(m.patient_id)
  ));
CREATE POLICY medication_schedule_update_primary ON public.medication_schedule
  FOR UPDATE TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.medication m
    WHERE m.medication_id = medication_schedule.medication_id
      AND public.is_primary_caregiver_for(m.patient_id)
  ))
  WITH CHECK (EXISTS (
    SELECT 1 FROM public.medication m
    WHERE m.medication_id = medication_schedule.medication_id
      AND public.is_primary_caregiver_for(m.patient_id)
  ));
CREATE POLICY medication_schedule_delete_primary ON public.medication_schedule
  FOR DELETE TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.medication m
    WHERE m.medication_id = medication_schedule.medication_id
      AND public.is_primary_caregiver_for(m.patient_id)
  ));

CREATE POLICY medication_log_select_active ON public.medication_log
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY medication_log_insert_assigned ON public.medication_log
  FOR INSERT TO authenticated
  WITH CHECK (
    caregiver_id = (SELECT auth.uid())
    AND public.can_record_care_for(patient_id, 'medication')
    AND EXISTS (
      SELECT 1 FROM public.medication m
      WHERE m.medication_id = medication_log.medication_id
        AND m.patient_id = medication_log.patient_id
    )
  );

CREATE POLICY daily_care_log_select_active ON public.daily_care_log
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY daily_care_log_insert_assigned ON public.daily_care_log
  FOR INSERT TO authenticated
  WITH CHECK (
    caregiver_id = (SELECT auth.uid())
    AND public.can_record_care_for(patient_id, type)
  );
CREATE POLICY time_event_select_active ON public.time_event
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY time_event_insert_assigned ON public.time_event
  FOR INSERT TO authenticated
  WITH CHECK (
    caregiver_id = (SELECT auth.uid())
    AND public.can_record_care_for(patient_id, event_type)
  );

CREATE POLICY scenario_card_select_authorized ON public.scenario_card
  FOR SELECT TO authenticated
  USING (patient_id IS NULL OR public.is_active_caregiver_for(patient_id));
CREATE POLICY scenario_option_select_authorized ON public.scenario_option
  FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.scenario_card sc
    WHERE sc.scenario_id = scenario_option.scenario_id
      AND (sc.patient_id IS NULL OR public.is_active_caregiver_for(sc.patient_id))
  ));

CREATE OR REPLACE FUNCTION public.guard_scenario_attempt_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
DECLARE
  scenario_patient_id uuid;
  correct_answer boolean;
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'Authentication is required';
  END IF;
  SELECT sc.patient_id, so.is_correct
  INTO scenario_patient_id, correct_answer
  FROM public.scenario_card sc
  JOIN public.scenario_option so ON so.scenario_id = sc.scenario_id
  WHERE sc.scenario_id = NEW.scenario_id
    AND so.option_id = NEW.option_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'The selected option does not belong to this scenario';
  END IF;
  IF scenario_patient_id IS NOT NULL
     AND NOT public.is_active_caregiver_for(scenario_patient_id) THEN
    RAISE EXCEPTION 'You are not authorized to attempt this scenario';
  END IF;
  NEW.caregiver_id := auth.uid();
  NEW.is_correct := correct_answer;
  NEW.attempted_at := COALESCE(NEW.attempted_at, now());
  RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS scenario_attempt_guard ON public.scenario_attempt;
CREATE TRIGGER scenario_attempt_guard
  BEFORE INSERT ON public.scenario_attempt
  FOR EACH ROW EXECUTE FUNCTION public.guard_scenario_attempt_insert();

CREATE POLICY scenario_attempt_select_own ON public.scenario_attempt
  FOR SELECT TO authenticated USING (caregiver_id = (SELECT auth.uid()));
CREATE POLICY scenario_attempt_insert_authorized ON public.scenario_attempt
  FOR INSERT TO authenticated
  WITH CHECK (
    caregiver_id = (SELECT auth.uid())
    AND EXISTS (
      SELECT 1 FROM public.scenario_card sc
      WHERE sc.scenario_id = scenario_attempt.scenario_id
        AND (sc.patient_id IS NULL OR public.is_active_caregiver_for(sc.patient_id))
    )
  );
REVOKE INSERT, UPDATE, DELETE ON public.scenario_attempt FROM authenticated;
GRANT INSERT (scenario_id, option_id, attempted_at) ON public.scenario_attempt TO authenticated;

CREATE POLICY coordination_metric_select_active ON public.coordination_metric
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));

CREATE POLICY emergency_contact_select_active ON public.emergency_contact
  FOR SELECT TO authenticated USING (public.is_active_caregiver_for(patient_id));
CREATE POLICY emergency_contact_insert_primary ON public.emergency_contact
  FOR INSERT TO authenticated WITH CHECK (public.is_primary_caregiver_for(patient_id));
CREATE POLICY emergency_contact_update_primary ON public.emergency_contact
  FOR UPDATE TO authenticated
  USING (public.is_primary_caregiver_for(patient_id))
  WITH CHECK (public.is_primary_caregiver_for(patient_id));
CREATE POLICY emergency_contact_delete_primary ON public.emergency_contact
  FOR DELETE TO authenticated USING (public.is_primary_caregiver_for(patient_id));

CREATE POLICY caregiver_select_self_or_circle ON public.caregiver
  FOR SELECT TO authenticated
  USING (
    caregiver_id = (SELECT auth.uid())
    OR EXISTS (
      SELECT 1
      FROM public.caregiver_pat mine
      JOIN public.caregiver_pat peer ON peer.patient_id = mine.patient_id
      WHERE mine.caregiver_id = (SELECT auth.uid())
        AND peer.caregiver_id = caregiver.caregiver_id
        AND mine.status = 'active' AND mine.removed_at IS NULL
        AND peer.status = 'active' AND peer.removed_at IS NULL
    )
  );
CREATE POLICY caregiver_update_self ON public.caregiver
  FOR UPDATE TO authenticated
  USING (caregiver_id = (SELECT auth.uid()))
  WITH CHECK (caregiver_id = (SELECT auth.uid()));
CREATE POLICY caregiver_insert_self ON public.caregiver
  FOR INSERT TO authenticated
  WITH CHECK (caregiver_id = (SELECT auth.uid()));
REVOKE UPDATE ON public.caregiver FROM authenticated;
GRANT UPDATE (full_name, phone, language, updated_at) ON public.caregiver TO authenticated;

CREATE POLICY care_notification_select_recipient_active ON public.care_notification
  FOR SELECT TO authenticated
  USING (
    recipient_id = (SELECT auth.uid())
    AND public.is_active_caregiver_for(patient_id)
  );
CREATE POLICY care_notification_update_read_state ON public.care_notification
  FOR UPDATE TO authenticated
  USING (
    recipient_id = (SELECT auth.uid())
    AND public.is_active_caregiver_for(patient_id)
  )
  WITH CHECK (
    recipient_id = (SELECT auth.uid())
    AND public.is_active_caregiver_for(patient_id)
  );
REVOKE INSERT, UPDATE, DELETE ON public.care_notification FROM authenticated;
GRANT UPDATE (read_at) ON public.care_notification TO authenticated;

COMMIT;

-- Scenario learning attempts are stored in scenario_attempt only. The ML API
-- intentionally excludes that value from care-coordination scoring.
